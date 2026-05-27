from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import json

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Lanna Detector API is running!"}



# ✅ เปิด CORS เพื่อให้ frontend เข้าถึงได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lanndetector-frontend.vercel.app", "http://localhost:3000"],  # ปรับเป็น domain ของ frontend ถ้ามี
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#✅ เปิด CORS เพื่อให้ fastapi เข้าถึงได้

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lanndetector-frontend.vercel.app", "http://localhost:3000"],  # หรือใส่เฉพาะ domain Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ โหลดโมเดล YOLOv8 ที่เทรนแล้ว
model = YOLO("best.pt")

# ✅ โหลด label ภาษาไทยล้านนา
with open("lan_labels.json", "r", encoding="utf-8") as f:
    lan_labels = json.load(f)

@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # ✅ รันโมเดล
        detection = model(image)[0]

        # ✅ แปลงผลลัพธ์เป็น JSON พร้อม label ภาษาไทย
        image_result = []
        for box in detection.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = lan_labels.get(str(cls_id), f"คลาส {cls_id}")

            image_result.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

        results.append(image_result)

    return {"results": results}