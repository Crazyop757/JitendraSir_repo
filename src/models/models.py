
"""
models.py

- Provides pretrained model loading for ResNet, Inception, ViT, EfficientNet, DenseNet, RegNet, Swin Transformer, ConvNeXt, MobileNet, and hybrid architectures
- Allows for custom classifier heads
- Supports loading with pretrained ImageNet weights
"""


import torch
import torch.nn as nn
from torchvision import models

from torchvision.models import (
    ResNet50_Weights, Inception_V3_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights,
    DenseNet121_Weights, RegNet_Y_400MF_Weights, Swin_V2_T_Weights, ConvNeXt_Tiny_Weights, MobileNet_V3_Large_Weights
)

# Dictionary of supported models

def get_model(model_name, num_classes, pretrained=True):
    def unfreeze_all(model):
        for param in model.parameters():
            param.requires_grad = True

    if model_name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    elif model_name == 'inception_v3':
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        # aux_logits must be True if using pretrained weights
        aux_logits = True if pretrained else False
        model = models.inception_v3(weights=weights, aux_logits=aux_logits)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    elif model_name == 'efficientnet_b0':
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    elif model_name == 'vit_b_16':
        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    elif model_name == 'densenet121':
        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    elif model_name == 'regnet_y_400mf':
        weights = RegNet_Y_400MF_Weights.DEFAULT if pretrained else None
        model = models.regnet_y_400mf(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    elif model_name == 'swin_v2_t':
        weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
        model = models.swin_v2_t(weights=weights)
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    elif model_name == 'convnext_tiny':
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    elif model_name == 'mobilenet_v3_large':
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        unfreeze_all(model)
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return model

# Example usage:
# model = get_model('resnet50', num_classes=4, pretrained=True)
