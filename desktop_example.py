from picamera2 import Picamera2
import cv2
import time
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
from collections import deque

# Rolling windows for 10-sample average
frame_times = deque(maxlen=25)
haar_times = deque(maxlen=25)
inference_times = deque(maxlen=25)

def rolling_avg(times):
    return sum(times) / len(times) if times else 0.0


# Load model and feature extractor
model_name = "mobilenet_v2_affectnethq-fer2013_model_fixed_labels"
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
    cap = cv2.VideoCapture(8)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    prev_time = time.time()
    fps = 0.0
    alpha = 0.1
    frame_count = 0

    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from USB camera.")
            break

        frame_read_time = time.time() - frame_start
        frame_times.append(frame_read_time)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Haar cascade every 3 frames
        faces = []
        if frame_count % 1 == 0:
            haar_start = time.time()
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
            haar_times.append(time.time() - haar_start)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            inputs = transform_inference(face_pil)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            inf_start = time.time()
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = outputs.logits.argmax(dim=-1).item()
            inference_times.append(time.time() - inf_start)

            emotion = label_mapping.get(prediction, "Unknown")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        frame_count += 1

        cv2.putText(frame, f"Frame: {rolling_avg(frame_times)*1000:.1f} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Haar: {rolling_avg(haar_times)*1000:.1f} ms", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Infer: {rolling_avg(inference_times)*1000:.1f} ms", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("USB Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
