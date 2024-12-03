from ultralytics import YOLO

# Load an official or custom model
# model = YOLO("yolo11n.pt")  # Load an official Detect model
model = YOLO("yolo11n-seg.pt")  # Load an official Segment model

# Perform tracking with the model
results = model.track(
    "videos/vid1.mp4", show=True, save=True
)  # Tracking with default tracker
