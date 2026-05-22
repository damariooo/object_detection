from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

model = YOLO("best.pt")

@app.get("/")
def home():
    return {"status": "AI RUNNING"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0])
            })

    return {
        "status": "success",
        "detections": detections
    }