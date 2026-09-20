import os
import json
import torch
import tempfile
from fastapi import FastAPI, UploadFile, File
import uvicorn
from src.step7_prediction import PlantDiseasePredictor

app = FastAPI(title="🌿 Plant Disease Detection API")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_file = 'models/class_names.json'
ckpt_file = 'models/best_model.pth'

predictor = None
if os.path.exists(class_file) and os.path.exists(ckpt_file):
    with open(class_file, 'r') as f:
        class_names = json.load(f)
    predictor = PlantDiseasePredictor(ckpt_file, class_names, device=device)

@app.get("/health")
def health():
    return {"status": "ok", "ready": predictor is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if predictor is None:
        return {"error": "Modèle non chargé. Lance l\'entraînement d\'abord."}
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    try:
        return predictor.predict(tmp_path)
    finally:
        os.unlink(tmp_path)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
