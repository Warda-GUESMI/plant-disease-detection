import os
import time
import copy
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(self, model, train_loader, val_loader, device='cuda', lr_head=1e-3, lr_backbone=1e-4, weight_decay=1e-4, label_smoothing=0.1, save_dir='models'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.optimizer = self._create_optimizer(head_only=True)
        self.scaler = GradScaler(enabled=(device == 'cuda'))

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
        self.best_val_acc = 0.0
        self.best_state = None

    def _create_optimizer(self, head_only: bool):
        head_params = list(self.model.head.parameters())
        if head_only:
            return optim.AdamW(head_params, lr=self.lr_head, weight_decay=self.weight_decay)
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        return optim.AdamW([
            {'params': backbone_params, 'lr': self.lr_backbone},
            {'params': head_params, 'lr': self.lr_head * 0.5}
        ], weight_decay=self.weight_decay)

    def _train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(self.train_loader, desc='  Train', leave=False, ncols=90):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(self.device == 'cuda')):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

        return running_loss / total, 100. * correct / total

    @torch.no_grad()
    def _validate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for images, labels in tqdm(self.val_loader, desc='  Val  ', leave=False, ncols=90):
            images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            with autocast(enabled=(self.device == 'cuda')):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / total
        val_acc = 100. * correct / total
        val_f1 = f1_score(all_labels, all_preds, average='macro') * 100
        return val_loss, val_acc, val_f1

    def fit(self, total_epochs=30, phase1_epochs=5, patience=7):
        logger.info("\n" + "=" * 65)
        logger.info("🏋️  STEP 5 : FINE-TUNING")
        logger.info("=" * 65)

        self.model.freeze_backbone()
        self.optimizer = self._create_optimizer(head_only=True)
        scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=phase1_epochs, eta_min=1e-6)
        patience_counter = 0

        for epoch in range(1, total_epochs + 1):
            if epoch == phase1_epochs + 1:
                logger.info("\n🔓 PHASE 2 : Fine-tuning du backbone...")
                self.model.unfreeze_backbone(from_layer=5)
                self.optimizer = self._create_optimizer(head_only=False)
                scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=1e-7)
                patience_counter = 0

            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc, val_f1 = self._validate()
            scheduler.step()
            current_lr = self.optimizer.param_groups[-1]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['lr'].append(current_lr)

            phase = "P1" if epoch <= phase1_epochs else "P2"
            logger.info(f"[{phase}] Epoch {epoch:2d}/{total_epochs} │ Train {train_loss:.4f}/{train_acc:.1f}% │ Val {val_loss:.4f}/{val_acc:.1f}% F1:{val_f1:.1f}% │ LR {current_lr:.1e}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                torch.save({'epoch': epoch, 'model_state_dict': self.best_state, 'val_acc': val_acc, 'history': self.history}, os.path.join(self.save_dir, 'best_model.pth'))
                logger.info(f"   ✅ Best model sauvegardé (Acc: {val_acc:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"\n⛔ Early Stopping à l\'epoch {epoch}")
                    break

        self.model.load_state_dict(self.best_state)
        return self.history

    def plot_history(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history['train_acc'], label='Train')
        axes[1].plot(epochs, self.history['val_acc'], label='Val')
        axes[1].set_title('Accuracy (%)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history['lr'])
        axes[2].set_title('Learning Rate'); axes[2].set_yscale('log'); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
