import cv2
import torch
import numpy as np
from torch.cuda.amp import autocast
from src.step3_transforms import get_val_transforms
from src.step4_model import build_model


class PlantDiseasePredictor:
    def __init__(self, checkpoint_path: str, class_names: list, img_size: int = 224,
                 device: str = 'cuda', confidence_threshold: float = 0.5):
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.confidence_threshold = confidence_threshold
        self.transform = get_val_transforms(img_size)

        self.model = build_model(num_classes=self.num_classes, pretrained=False, device=device)
        # weights_only=True : n'execute pas de code arbitraire contenu dans le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, image_path: str, top_k: int = 5) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image introuvable ou illisible : {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.predict_array(image_rgb, top_k=top_k)

    @torch.no_grad()
    def predict_array(self, image_rgb: np.ndarray, top_k: int = 5) -> dict:
        """Predit a partir d'une image RGB (H, W, 3, uint8) deja en memoire."""
        tensor = self.transform(image=image_rgb)['image'].unsqueeze(0).to(self.device)
        with autocast(enabled=(self.device == 'cuda')):
            output = self.model(tensor)

        probs = torch.softmax(output.float(), dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        pred_class = self.class_names[pred_idx]
        parts = pred_class.split('___') if '___' in pred_class else [pred_class, 'unknown']

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_k_results = [{'class': self.class_names[i], 'probability': float(probs[i])} for i in top_indices]
        confidence = float(probs[pred_idx])

        return {
            'plant': parts[0],
            'disease': parts[1] if len(parts) > 1 else pred_class,
            'confidence': confidence,
            'is_healthy': 'healthy' in pred_class.lower(),
            # True si la confiance est sous le seuil : l'image est peut-etre hors sujet
            'is_uncertain': confidence < self.confidence_threshold,
            'top_k': top_k_results
        }
