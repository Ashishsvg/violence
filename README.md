🚨 Violence Detection Using YOLOv8 (ONGOING)

Real-time violence detection system built using Python, YOLOv8, and OpenCV, designed to identify violent activities such as fights or aggressive movements in video streams.

📌 Overview

This project aims to detect violent actions in real-time using a deep learning–based object detection model (YOLOv8).
The system processes video frames, runs inference on each frame, and triggers alerts when violence is detected.

This is my final year major project (ongoing), focusing on:

Computer Vision

Deep Learning

Object Detection

Real-time video analytics

🎯 Project Objectives

Build a robust model to detect violence in video footage

Train YOLOv8 on a custom annotated dataset

Achieve high accuracy with minimal false positives

Integrate the model with a real-time video pipeline

Display bounding boxes + class labels during detection

Future extension: Alert system (SMS/Email/API trigger)

🛠️ Tech Stack

Python 3.10+

YOLOv8 (Ultralytics)

OpenCV

NumPy / Pandas

Jupyter Notebook

PyTorch (backend engine)

📂 Project Structure
📁 Violence-Detection-YOLOv8
│── data/               # Dataset (images + labels)
│── models/             # Trained YOLOv8 weights
│── notebooks/          # Training notebooks
│── src/
│    ├── train.py       # YOLOv8 model training
│    ├── detect.py      # Real-time violence detection script
│    ├── utils.py       # Helper functions
│── README.md           # Documentation
│── requirements.txt    # Dependencies

🚀 How It Works
1️⃣ Model Training

Dataset is annotated using LabelImg / Roboflow

Labels include actions like:

fight, punch, kick, weapon, violence

Training is done using:

yolo train model=yolov8s.pt data=data.yaml epochs=50 imgsz=640

2️⃣ Real-Time Detection

The system uses OpenCV to capture video frames and passes each frame to YOLOv8 for prediction.

from ultralytics import YOLO
import cv2

model = YOLO("models/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model.predict(frame)
    annotated = results[0].plot()
    cv2.imshow("Violence Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

🧪 Features

✔ Real-time violence detection
✔ Bounding boxes + labels
✔ Custom YOLOv8 model
✔ Works with webcam or video files
✔ Custom training pipeline
✔ Modular & extendable code

📊 Model Performance (Ongoing)
Metric	Value
mAP50	Coming soon
Accuracy	Coming soon
F1 Score	Coming soon

(Will be updated once training is completed.)

🔮 Future Enhancements

Add alert notification system (email / SMS / webhook)

Deploy using Streamlit or Flask

Build a CCTV/dashboard UI

Train on larger datasets to improve accuracy

Add violence severity scoring

🧑‍💻 Author

Ashish Tangde
Python Developer | ML Enthusiast
📧 Email: your-email
🔗 LinkedIn: your-link
🐙 GitHub: your-username

⭐ Like this project?

If you found this useful, consider giving it a ⭐ star on GitHub!
