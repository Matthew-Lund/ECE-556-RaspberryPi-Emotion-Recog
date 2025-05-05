# ECE-556-RaspberryPi-Emotion-Recog
WPI ECE 556 Raspberry Pi Emotion Recognition Deep Learning Project

# How to run code:

Demo example was run on Raspberry Pi 5 with a IMX708 camera module. Cloning the repository with the correct hardware configuration (Camera module in cam1 location) will allow you to run the code. A window will open with the camera feed and model inference overlay, displaying the frame time and model inference time.
1. Install python requirements (torch, transformers, picamera2, opencv-python)
2. python3 original_example.py

# Architecture Diagram:


# Model Performance Analysis

Validation Loss:

![Validation Loss Comparison](validation_loss_comparison.png)

Accuracy:

![Accuracy Comparison](accuracy_comparison.png)

Precision:

![Precision Comparison](precision_comparison.png)

Recall:
![Recall Comparison](recall_comparison.png)

F1:

![F1 Comparison](f1_comparison.png)
