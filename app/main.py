from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from ultralytics import YOLO
from PIL import Image
import io
import torch


MODEL_PATH = "./weights/best.pt"
CONFIDENCE_THRESHOLD = 0.5


# --- Pydantic модели (из контракта API) ---
class BoundingBox(BaseModel):
    """Абсолютные координаты BoundingBox"""
    x_min: int = Field(..., description="Левая координата", ge=0)
    y_min: int = Field(..., description="Верхняя координата", ge=0)
    x_max: int = Field(..., description="Правая координата", ge=0)
    y_max: int = Field(..., description="Нижняя координата", ge=0)


class Detection(BaseModel):
    """Результат детекции одного логотипа"""
    bbox: BoundingBox = Field(..., description="Результат детекции")


class DetectionResponse(BaseModel):
    """Ответ API с результатами детекции"""
    detections: List[Detection] = Field(..., description="Список найденных логотипов")


class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Описание ошибки")
    detail: Optional[str] = Field(None, description="Дополнительная информация")


app = FastAPI(
    title="T-Bank Logo Detection API",
    description="API для обнаружения логотипа Т-Банка на изображениях.",
    version="1.0.0"
)

# Проверяем доступность GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model on device: {device.upper()}")


try:
    model = YOLO(MODEL_PATH)
    model.to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


@app.post(
    "/detect",
    response_model=DetectionResponse,
    responses={500: {"model": ErrorResponse}}
)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-Банка на изображении
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")

    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/webp"]:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid file format", "detail": "Supported formats: JPEG, PNG, BMP, WEBP"}
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": "Could not read image file", "detail": str(e)}
        )

    try:
        results = model.predict(source=image, conf=CONFIDENCE_THRESHOLD)

        detections = []
        for result in results:
            for box in result.boxes.xyxy:  # .xyxy возвращает тензор [xmin, ymin, xmax, ymax]
                x_min, y_min, x_max, y_max = [int(coord) for coord in box]
                detections.append(Detection(bbox=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)))

        return DetectionResponse(detections=detections)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "detail": str(e)}
        )
