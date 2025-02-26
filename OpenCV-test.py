from picamera2 import Picamera2
import cv2
import time

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    
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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw rectangle around faces
        
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
