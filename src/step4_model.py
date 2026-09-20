import torch
import torch.nn as nn
import torchvision.models as models

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b3(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=dropout * 0.6),
            nn.Linear(512, num_classes)
        )

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, from_layer: int = 5):
        for i, block in enumerate(self.backbone.features):
            if i >= from_layer:
                for param in block.parameters():
                    param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def build_model(num_classes: int, pretrained: bool = True, dropout: float = 0.3, device: str = 'cuda'):
    model = PlantDiseaseModel(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    return model.to(device)
