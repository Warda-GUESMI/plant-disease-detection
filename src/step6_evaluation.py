import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, top_k_accuracy_score

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, test_loader, class_names, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.labels, self.preds, self.probs = [], [], []
        self._collect_predictions()

    @torch.no_grad()
    def _collect_predictions(self):
        for images, labels in tqdm(self.test_loader, desc='  Test ', ncols=90):
            images = images.to(self.device, non_blocking=True)
            with autocast(enabled=(self.device == 'cuda')):
                outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            self.labels.extend(labels.numpy())
            self.preds.extend(preds)
            self.probs.extend(probs)

        self.labels = np.array(self.labels)
        self.preds = np.array(self.preds)
        self.probs = np.array(self.probs)

    def print_metrics(self):
        acc = accuracy_score(self.labels, self.preds)
        f1_macro = f1_score(self.labels, self.preds, average='macro')
        f1_weighted = f1_score(self.labels, self.preds, average='weighted')
        prec = precision_score(self.labels, self.preds, average='macro')
        rec = recall_score(self.labels, self.preds, average='macro')
        k_max = min(5, len(self.class_names))
        top3 = top_k_accuracy_score(self.labels, self.probs, k=min(3, k_max))
        top5 = top_k_accuracy_score(self.labels, self.probs, k=k_max)

        logger.info("\n" + "=" * 60)
        logger.info("📊 STEP 6 : RÉSULTATS D\'ÉVALUATION")
        logger.info("=" * 60)
        logger.info(f"   Accuracy        : {acc * 100:.2f}%")
        logger.info(f"   Macro F1        : {f1_macro * 100:.2f}%")
        logger.info(f"   Weighted F1     : {f1_weighted * 100:.2f}%")
        logger.info(f"   Precision (M)   : {prec * 100:.2f}%")
        logger.info(f"   Recall (M)      : {rec * 100:.2f}%")
        logger.info(f"   Top-3 Accuracy  : {top3 * 100:.2f}%")
        logger.info(f"   Top-5 Accuracy  : {top5 * 100:.2f}%")
        logger.info("=" * 60)

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.labels, self.preds)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        n = len(self.class_names)
        fig, ax = plt.subplots(figsize=(max(12, n * 0.5), max(12, n * 0.5)))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_xlabel('Prédiction')
        ax.set_ylabel('Vérité')
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')

    def full_evaluation(self):
        self.print_metrics()
        self.plot_confusion_matrix()
