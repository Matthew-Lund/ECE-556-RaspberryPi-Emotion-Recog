from picamera2 import Picamera2
import cv2
import time
from datasets import load_dataset
from transformers import AutoFeatureExtractor  , AutoModelForImageClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import torch

# Transform function for inference (single image)
def transform_inference(image):
    image = image.convert("RGB")  # Ensure RGB format
    inputs = feature_extractor(image, return_tensors="pt")  # Apply the same feature extractor
    return inputs

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    prev_time = time.time()
    fps = 0.0
    alpha = 0.1  # Smoothing factor for FPS
    frame_count = 0

    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    model = "mobilenet_v2_affectnethq-fer2013_model_fixed_labels"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model)
    model = AutoModelForImageClassification.from_pretrained(model).to(device)
    model.eval()
    
    label_mapping = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Angry", 4: "Surprise", 5: "Fear", 6: "Disgust"}  # Adjust based on model

    while True:
        frame = picam2.capture_array()  # Capture frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        
        if frame_count % 3 == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:

            face = frame[y:y+h, x:x+w]  # Crop face
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))  # Convert to PIL image
    
            inputs = transform_inference(face_pil)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = outputs.logits.argmax(dim=-1).item()
                emotion = label_mapping.get(prediction, "Unknown")
    
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around faces
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        frame_count = frame_count + 1
        
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
