from picamera2 import Picamera2
import cv2
import time
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image

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
    # Initialize camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (224, 224)}))
    picam2.start()
    
    # Load Haarcascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    prev_time = time.time()
    fps = 0.0
    alpha = 0.1  # Smoothing factor for FPS
    frame_count = 0

    while True:
        frame = picam2.capture_array()  # Capture frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
        if frame_count % 3 == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]  # Crop the face
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert to PIL image
            
            # Apply transformation
            inputs = transform_inference(face_pil)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to correct device
            
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = outputs.logits.argmax(dim=-1).item()
                emotion = label_mapping.get(prediction, "Unknown")

            # Draw rectangle and label on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        elapsed_time = current_time - prev_time
        prev_time = current_time
        
        if elapsed_time > 0:
            instant_fps = 1.0 / elapsed_time
            fps = (alpha * instant_fps) + (1 - alpha) * fps  # Apply smoothing
        
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Raspberry Pi Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    picam2.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
