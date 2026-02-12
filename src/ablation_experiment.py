"""
ablation_experiment.py

- Runs ablation studies for model, preprocessing, and optimization
- Reports quantitative and qualitative results for each ablation
"""

import torch
from src.models.models import get_model
from src.preprocessing.preprocessing import get_stratified_kfold, set_seed, get_train_transforms, get_val_transforms
from src.evaluation.evaluation import evaluate_predictions
from src.ablation.ablation import ablate_component
from sklearn.metrics import accuracy_score, f1_score

CONFIG = {
    'num_classes': 4,
    'epochs': 5,
    'batch_size': 32,
    'lr': 1e-4,
    'optimizer': 'adam',
    'loss_fn': 'cross_entropy',
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': 'archive/lung_colon_image_set',
    'n_splits': 3,
}

def train_eval(model, data, train_args):
    train_dataset, val_dataset, class_names = data
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_args['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=train_args['batch_size'], shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_args['lr'])
    model.to(train_args['device'])
    for epoch in range(train_args['epochs']):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(train_args['device']), labels.to(train_args['device'])
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(train_args['device']), labels.to(train_args['device'])
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"Ablation result: Accuracy={acc:.4f}, Macro F1={f1:.4f}")
    evaluate_predictions(y_true, y_pred, class_names)
    return {'acc': acc, 'f1': f1}

def main():
    set_seed(CONFIG['seed'])
    ablation_configs = {
        'resnet50_default_aug': {
            'model_args': {'model_name': 'resnet50', 'num_classes': CONFIG['num_classes'], 'pretrained': True},
            'train_args': CONFIG.copy(),
            'preprocessing': get_train_transforms
        },
        'efficientnet_no_aug': {
            'model_args': {'model_name': 'efficientnet_b0', 'num_classes': CONFIG['num_classes'], 'pretrained': True},
            'train_args': CONFIG.copy(),
            'preprocessing': get_val_transforms
        },
        'densenet_with_aug': {
            'model_args': {'model_name': 'densenet121', 'num_classes': CONFIG['num_classes'], 'pretrained': True},
            'train_args': CONFIG.copy(),
            'preprocessing': get_train_transforms
        },
    }
    for ablation_name, config in ablation_configs.items():
        print(f"\n--- Running ablation: {ablation_name} ---")
        for fold, train_dataset, val_dataset, class_names in get_stratified_kfold(CONFIG['data_dir'], CONFIG['n_splits'], CONFIG['seed']):
            # Apply ablation-specific preprocessing
            train_dataset.dataset.transform = config['preprocessing']()
            val_dataset.dataset.transform = get_val_transforms()
            data = (train_dataset, val_dataset, class_names)
            model = get_model(**config['model_args'])
            train_eval(model, data, config['train_args'])

if __name__ == "__main__":
    main()
