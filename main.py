from fastapi import FastAPI
from ultralytics import YOLO

app = FastAPI()

model = None

@app.on_event("startup")
def startup_event():
    global model

    print("LOADING MODEL...")

    model = YOLO("best.pt")

    print("MODEL LOADED")

@app.get("/")
def home():
    return {"status": "RUNNING"}