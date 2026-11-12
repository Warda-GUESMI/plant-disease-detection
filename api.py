import os
import json
import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.step7_prediction import PlantDiseasePredictor

app = FastAPI(title="Plant Disease Detection API")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_file = 'models/class_names.json'
ckpt_file = 'models/best_model.pth'

predictor = None
if os.path.exists(class_file) and os.path.exists(ckpt_file):
    with open(class_file, 'r', encoding='utf-8') as f:
        class_names = json.load(f)
    predictor = PlantDiseasePredictor(ckpt_file, class_names, device=device,
                                      confidence_threshold=CONFIDENCE_THRESHOLD)


@app.get("/health")
def health():
    return {"status": "ok", "ready": predictor is not None}


# Endpoint synchrone : FastAPI l'execute dans un thread, l'inference ne bloque pas la boucle d'evenements.
@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Modele non charge. Lancez l'entrainement d'abord.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Format non supporte : utilisez JPEG ou PNG.")

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (max {MAX_UPLOAD_BYTES // (1024 * 1024)} Mo).")

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Image illisible ou corrompue.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return predictor.predict_array(image_rgb)


if __name__ == '__main__':
    # 127.0.0.1 par defaut : n'exposez l'API sur le reseau (API_HOST=0.0.0.0) que derriere un proxy/auth.
    uvicorn.run(app, host=os.getenv("API_HOST", "127.0.0.1"), port=int(os.getenv("API_PORT", "8000")))
