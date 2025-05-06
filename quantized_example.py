from picamera2 import Picamera2
import cv2
import time
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from collections import deque

# Rolling windows for 10-sample average
frame_times = deque(maxlen=25)
inference_times = deque(maxlen=25)

# Rolling window for smoothed predictions (e.g., last 5 predictions)
prediction_window = deque(maxlen=10)

def rolling_avg(times):
    return sum(times) / len(times) if times else 0.0

# Load model and feature extractor
model_name = "mobilenet_v2_affectnethq-fer2013_quantized_pruned"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForImageClassification.from_pretrained(
    model_name, ignore_mismatched_sizes=True
)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Emotion label mapping (adjust based on your model)
label_mapping = {6: "Neutral", 3: "Happy", 4: "Sad", 0: "Angry", 5: "Surprise", 2: "Fear", 1: "Disgust"}

# Define the transformation function for inference
def transform_inference(image):
    image = image.convert("RGB")  # Ensure it's in RGB format
    inputs = feature_extractor(image, return_tensors="pt")  # Apply the feature extractor
    return inputs

def main():
    # Initialize camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    prev_time = time.time()
    fps = 0.0
    alpha = 0.1  # Smoothing factor for FPS
    frame_count = 0

    while True:
        frame_start = time.time()
        frame = picam2.capture_array()  # Capture frame
        frame_read_time = time.time() - frame_start
        frame_times.append(frame_read_time)
        
        # Optional: crop center of frame (simulate assumed face region)
        h, w, _ = frame.shape
        crop_w, crop_h = int(w * 0.6), int(h * 0.6)
        x1 = (w - crop_w) // 2
        y1 = (h - crop_h) // 2
        face = frame[y1:y1+crop_h, x1:x1+crop_w]

        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Only run inference every N frames
        N = 5  # Run inference every 5 frames
        if frame_count % N == 0:
            inputs = transform_inference(face_pil)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            inf_start = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = outputs.logits.argmax(dim=-1).item()
            inference_times.append(time.time() - inf_start)

            # Add prediction to window for smoothing
            prediction_window.append(prediction)

            # Get the most common prediction in the window
            most_common_prediction = max(set(prediction_window), key=prediction_window.count)
            emotion = label_mapping.get(most_common_prediction, "Unknown")
            
        elif prediction_window:
            most_common_prediction = max(set(prediction_window), key=prediction_window.count)
            emotion = label_mapping.get(most_common_prediction, "Unknown")

        # Show timing info
        cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.putText(frame, f"Frame: {rolling_avg(frame_times)*1000:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Infer: {rolling_avg(inference_times)*1000:.1f} ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("IMX708 Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break
            
        frame_count += 1

    picam2.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
