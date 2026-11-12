# 🌿 Plant Disease Detection using Deep Learning

An AI-powered Computer Vision project that detects plant diseases from leaf images using **PyTorch and EfficientNet-B3**.

### Key Features
- Plant and disease classification (target: the 38 classes of PlantVillage).
- Transfer Learning with EfficientNet-B3 (two-phase fine-tuning).
- Image cleaning, near-duplicate removal, preprocessing and data augmentation.
- Model training and evaluation (Accuracy, F1-score, Confusion Matrix).
- Disease prediction with confidence scores and a low-confidence flag.
- REST API using FastAPI.
- Interactive web interface using Streamlit.

### Tech Stack
Python · PyTorch · EfficientNet-B3 · OpenCV · Scikit-learn · FastAPI · Streamlit · NumPy · Pandas

### Dataset
PlantVillage – approximately 54,000 images across 38 classes.

> **Current status of the committed results.** The files in `results/` and `models/class_names.json`
> come from a **6-class subset** (Apple scab / black rot / cedar rust / healthy, Blueberry healthy,
> Cherry healthy; about 10,000 images) trained for **5 epochs with the backbone frozen**
> (phase 1 only). Re-run the full pipeline on all 38 classes before quoting any figure.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Put the dataset in `data/PlantVillage/<class_name>/*.jpg` (or use `python run_all.py`, which extracts
`~/Downloads/archive.zip` for you), then:

```bash
python main.py --step all --epochs 30 --batch_size 32   # clean -> split -> train -> eval
python main.py --step predict --image path/to/leaf.jpg
```

Useful options: `--step {all,clean,split,train,eval,predict}`, `--epochs`, `--phase1` (frozen-backbone
epochs, then the backbone is unfrozen), `--img_size`, `--seed` (default 42).

Trained weights are not committed (`models/*.pth` is git-ignored). `models/best_model.pth` and
`models/class_names.json` are produced by the training step and must match each other.

### REST API

```bash
python api.py                       # http://127.0.0.1:8000  (docs at /docs)
curl -F "file=@leaf.jpg;type=image/jpeg" http://127.0.0.1:8000/predict
```

| Variable | Default | Meaning |
|---|---|---|
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | Bind address. Only use `0.0.0.0` behind a reverse proxy with authentication. |
| `MAX_UPLOAD_MB` | `5` | Maximum upload size (HTTP 413 above). |
| `CONFIDENCE_THRESHOLD` | `0.5` | Below it, the response has `"is_uncertain": true`. |

Errors: `503` model not loaded, `415` not JPEG/PNG, `413` file too large, `400` unreadable image.

### Web app

```bash
streamlit run app_web.py
```

### Tests

```bash
pip install -r requirements-dev.txt fastapi python-multipart opencv-python-headless
python -m pytest
```

## Known limitations

- PlantVillage images are lab photos on plain backgrounds: expect a clear drop in accuracy on field photos.
- The random split can leave near-identical images (especially in augmented versions of the dataset) in
  both train and test, which inflates the metrics. A group-aware split is recommended.
- The model has no "not a leaf" class: use the confidence flag, but it is not a reliable out-of-distribution detector.
- This is an educational project, not a substitute for an agronomist's diagnosis.

**Domain:** Deep Learning | Computer Vision | Artificial Intelligence | Smart Agriculture
