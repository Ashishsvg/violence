

from ultralytics import YOLO
import cv2
import os

dataset_path = "Dataset"
os.makedirs(os.path.join(dataset_path, "Violence"), exist_ok=True)
os.makedirs(os.path.join(dataset_path, "Non_Violence"), exist_ok=True)

print("📁 Dataset folder structure ready:")
print(f"{dataset_path}/")
print(" ┣ Violence/")
print(" ┗ Non_Violence/")


model = YOLO("yolov8n.pt") 
print("✅ Pretrained YOLOv8 model loaded.")


video_path = "v3.mp4"  

if not os.path.exists(video_path):
    print("⚠️ Please add a video named 'test.mp4' in your project folder.")
else:
    cap = cv2.VideoCapture(video_path)
    print("🎥 Running object detection on video... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)
        annotated_frame = results[0].plot()

        # Display video with detections
        cv2.imshow("YOLOv8 Object Detection (Demo)", annotated_frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Demo complete: Objects detected in video.")
