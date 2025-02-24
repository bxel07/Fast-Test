import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import easyocr

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
yolo_model = YOLO("best.pt")  

# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(["id"])  

# Directory for saving cropped images
OUTPUT_DIR = "cropped_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/read-text")
async def detect_and_read(file: UploadFile = File(...)):
    """
    Detect objects using YOLO, extract detected areas, apply OCR, and return results.
    
    Args:
        file: Uploaded image file

    Returns:
        JSON with detected objects, extracted text, and saved cropped images.
    """
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    results = yolo_model(image)

    # Initialize result list
    detections = []

    # Process detection results
    for result in results:
        boxes = result.boxes  # Bounding box coordinates
        
        for i, box in enumerate(boxes):
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Extract detected region (crop)
            cropped_img = image[y1:y2, x1:x2]

            # Save cropped image
            cropped_filename = f"{OUTPUT_DIR}/detected_{i}.jpg"
            cv2.imwrite(cropped_filename, cropped_img)

            # Perform OCR on cropped region
            ocr_results = ocr_reader.readtext(cropped_img)

            # Extract text from OCR results
            text_detections = []
            for bbox, text, prob in ocr_results:
                bbox_fixed = [[int(x), int(y)] for (x, y) in bbox]
                text_detections.append({
                    "text": text,
                    "confidence": prob,
                    "bbox": bbox_fixed
                })

            # Store detection results
            detections.append({
                "class_name": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [x1, y1, x2, y2],
                "cropped_image": cropped_filename,
                "text_detections": text_detections
            })

    return {"detections": detections}


@app.post("/detect-objects")
async def detect_objects(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Run YOLOv8 inference
    results = yolo_model(img)
    
    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].tolist()  # get box coordinates
            conf = float(box.conf)
            cls = int(box.cls)
            name = results[0].names[cls]
            detections.append({
                "bbox": b,
                "confidence": conf,
                "class": name
            })
    
    return {"detections": detections}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)