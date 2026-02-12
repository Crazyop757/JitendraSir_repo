
"""
models.py

- Provides pretrained model loading for ResNet, Inception, ViT, EfficientNet, DenseNet, RegNet, Swin Transformer, ConvNeXt, MobileNet, SE-ResNet, and hybrid architectures
- Allows for custom classifier heads
- Supports loading with pretrained ImageNet weights
"""


import torch
import torch.nn as nn
from torchvision import models

from torchvision.models import (
    ResNet50_Weights, Inception_V3_Weights, EfficientNet_B0_Weights, ViT_B_16_Weights,
    DenseNet121_Weights, RegNet_Y_400MF_Weights, Swin_V2_T_Weights, ConvNeXt_Tiny_Weights, 
    MobileNet_V3_Large_Weights, EfficientNet_B4_Weights
)

# Try to import timm for SE-ResNet and other advanced models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm not installed. SE-ResNet will not be available. Install with: pip install timm")


# ==================== Squeeze-and-Excitation Block ====================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ==================== Hybrid CNN-ViT Model ====================
class HybridCNNViT(nn.Module):
    """
    Hybrid CNN + Vision Transformer model
    - Uses CNN (ResNet) as feature extractor
    - Feeds CNN features into a lightweight Transformer encoder
    - Combines local (CNN) and global (Transformer) features
    """
    def __init__(self, num_classes, pretrained=True):
        super(HybridCNNViT, self).__init__()
        
        # CNN backbone (ResNet50 without final FC)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])  # Remove avgpool and fc
        
        # CNN output: 2048 channels, 7x7 spatial (for 224x224 input)
        self.cnn_channels = 2048
        self.patch_size = 7  # 7x7 = 49 patches
        
        # Project CNN features to transformer dimension
        self.embed_dim = 512
        self.proj = nn.Conv2d(self.cnn_channels, self.embed_dim, kernel_size=1)
        
        # Positional embedding for 49 patches + 1 class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_size * self.patch_size + 1, self.embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=8, 
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.embed_dim, num_classes)
        )
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # CNN feature extraction: (B, 2048, 7, 7)
        x = self.cnn_backbone(x)
        
        # Project to embed_dim: (B, 512, 7, 7)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, 512, 49) -> (B, 49, 512)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token: (B, 50, 512)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Use class token for classification
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x


# ==================== Main Model Factory ====================
def get_model(model_name, num_classes, pretrained=True):
    """
    Factory function to get pretrained models with custom classification heads.
    
    Supported models:
    - resnet50: ResNet-50
    - densenet121: DenseNet-121
    - efficientnet_b0: EfficientNet-B0
    - efficientnet_b4: EfficientNet-B4
    - vit_b_16: Vision Transformer Base (patch 16)
    - swin_v2_t: Swin Transformer V2 Tiny
    - se_resnet50: SE-ResNet-50 (Squeeze-and-Excitation)
    - hybrid_cnn_vit: Hybrid CNN + Vision Transformer
    - inception_v3: Inception V3
    - regnet_y_400mf: RegNet Y 400MF
    - convnext_tiny: ConvNeXt Tiny
    - mobilenet_v3_large: MobileNet V3 Large
    """
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
        
    elif model_name == 'efficientnet_b4':
        weights = EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b4(weights=weights)
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
        
    elif model_name == 'se_resnet50':
        # SE-ResNet using timm library
        if TIMM_AVAILABLE:
            model = timm.create_model('seresnet50', pretrained=pretrained, num_classes=num_classes)
            # Add dropout before final classifier
            if hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, num_classes)
                )
            unfreeze_all(model)
        else:
            raise ValueError("SE-ResNet requires timm library. Install with: pip install timm")
            
    elif model_name == 'hybrid_cnn_vit':
        # Custom Hybrid CNN + Vision Transformer
        model = HybridCNNViT(num_classes=num_classes, pretrained=pretrained)
        unfreeze_all(model)
        
    else:
        raise ValueError(f"Model {model_name} not supported. Available models: resnet50, densenet121, "
                        f"efficientnet_b0, efficientnet_b4, vit_b_16, swin_v2_t, se_resnet50, "
                        f"hybrid_cnn_vit, inception_v3, regnet_y_400mf, convnext_tiny, mobilenet_v3_large")
    
    return model


def get_model_input_size(model_name):
    """Returns the required input size for each model"""
    if model_name == 'inception_v3':
        return 299
    elif model_name == 'efficientnet_b4':
        return 380  # EfficientNet-B4 optimal size
    else:
        return 224  # Default for most models

# Example usage:
# model = get_model('resnet50', num_classes=4, pretrained=True)
