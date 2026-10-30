
from pathlib import Path

from ultralytics import YOLO

# Load the custom model from the project folder
MODEL_PATH = Path(__file__).parent / "lab_device_best.pt"
model = YOLO(str(MODEL_PATH))

# Map dataset class names to labels displayed in the app
DEVICE_LABELS = {
    "Cpu": "CPU",
    "Monitor": "Monitor",
    "Keyboard": "Keyboard",
    "Mouse": "Mouse",
}


def detect_devices(image, confidence=0.35, imgsz=640):
    """
    Detect and count computer lab devices in an image.

    Args:
        image: Image as a PIL image or NumPy array.
        confidence: Minimum confidence threshold.
        imgsz: Image size used for inference.

    Returns:
        annotated_image: Image with detection boxes.
        counts: Dictionary containing counts by device type.
        detections: List of individual detections.
    """
    results = model.predict(
        source=image,
        conf=confidence,
        imgsz=imgsz,
        iou=0.45,
        agnostic_nms=True,
        verbose=False,
    )

    result = results[0]
    counts = {label: 0 for label in DEVICE_LABELS.values()}
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]

            # Ignore classes that are not computer devices we count
            if class_name not in DEVICE_LABELS:
                continue

            label = DEVICE_LABELS[class_name]
            confidence_score = float(box.conf[0].item())

            x1, y1, x2, y2 = (
                int(value) for value in box.xyxy[0].tolist()
            )

            counts[label] += 1

            detections.append(
                {
                    "device": label,
                    "confidence": round(confidence_score, 3),
                    "box": [x1, y1, x2, y2],
                }
            )

    annotated_image = result.plot()

    return annotated_image, counts
