"""Tests de l'API avec un predicteur factice (sans torch ni checkpoint)."""
import sys
import types

import cv2
import numpy as np
import pytest

# Stub du module d'inference pour ne pas importer torch
stub = types.ModuleType("src.step7_prediction")


class _FakePredictor:
    def __init__(self, *a, **k):
        pass

    def predict_array(self, image_rgb, top_k=5):
        assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
        return {"plant": "Apple", "disease": "Apple_scab", "confidence": 0.9,
                "is_healthy": False, "is_uncertain": False, "top_k": []}


stub.PlantDiseasePredictor = _FakePredictor
sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules["src.step7_prediction"] = stub
sys.modules.setdefault("torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402


def _png_bytes():
    img = np.full((32, 32, 3), 120, dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture
def client():
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").status_code == 200


def test_503_when_model_missing(client, monkeypatch):
    monkeypatch.setattr(api, "predictor", None)
    r = client.post("/predict", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 503


def test_ok(client, monkeypatch):
    monkeypatch.setattr(api, "predictor", _FakePredictor())
    r = client.post("/predict", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 200 and r.json()["plant"] == "Apple"


def test_415_wrong_type(client, monkeypatch):
    monkeypatch.setattr(api, "predictor", _FakePredictor())
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_400_corrupt_image(client, monkeypatch):
    monkeypatch.setattr(api, "predictor", _FakePredictor())
    r = client.post("/predict", files={"file": ("a.png", b"not an image", "image/png")})
    assert r.status_code == 400


def test_413_too_large(client, monkeypatch):
    monkeypatch.setattr(api, "predictor", _FakePredictor())
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    r = client.post("/predict", files={"file": ("a.png", _png_bytes(), "image/png")})
    assert r.status_code == 413
