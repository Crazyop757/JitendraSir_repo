"""
explainability.py

- Integrates Grad-CAM, LIME, and SHAP for model explainability
- Provides functions for qualitative and quantitative analysis
"""

import torch
import numpy as np
from torchvision import transforms

# Grad-CAM (using pytorch-grad-cam)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None
    show_cam_on_image = None

# Grad-CAM
def run_gradcam(model, input_tensor, target_layer):
    if GradCAM is None:
        raise ImportError("pytorch-grad-cam not installed.")
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    return grayscale_cam

# LIME integration
def run_lime(model, image_np, class_names):
    from lime import lime_image
    explainer = lime_image.LimeImageExplainer()
    def batch_predict(images):
        model.eval()
        images = torch.stack([transforms.ToTensor()(img) for img in images], dim=0)
        with torch.no_grad():
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
        return probs
    explanation = explainer.explain_instance(
        image_np,
        batch_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    return explanation

# SHAP integration
def run_shap(model, images):
    import shap
    model.eval()
    background = images[:10]
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(images)
    return shap_values

# Case study function example
def case_study_success_error(model, dataloader, explain_fn, num_samples=5):
    """Run explain_fn on a few samples and print qualitative results."""
    model.eval()
    results = []
    count = 0
    for images, labels in dataloader:
        for i in range(len(images)):
            img = images[i].unsqueeze(0)
            label = labels[i].item()
            try:
                explanation = explain_fn(model, img)
                results.append({'image': img, 'label': label, 'explanation': explanation})
                print(f"Sample {count+1}: label={label}, explanation generated.")
            except Exception as e:
                print(f"Sample {count+1}: label={label}, explanation failed: {e}")
            count += 1
            if count >= num_samples:
                return results
    return results
