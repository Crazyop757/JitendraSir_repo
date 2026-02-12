"""
ablation.py

- Ablation study utilities
- Allows toggling model components, preprocessing steps, and optimization algorithms
- Reports quantitative and qualitative results for each ablation
"""

def ablate_component(model_fn, ablation_configs, train_fn, eval_fn, data):
    results = {}
    for name, config in ablation_configs.items():
        print(f"Running ablation: {name}")
        model = model_fn(**config['model_args'])
        train_fn(model, data, **config['train_args'])
        metrics = eval_fn(model, data)
        results[name] = metrics
    return results

# Example usage:
# ablation_configs = {
#     'no_augmentation': {'model_args': {...}, 'train_args': {...}},
#     'with_augmentation': {'model_args': {...}, 'train_args': {...}},
# }
# ablate_component(get_model, ablation_configs, train_fn, eval_fn, data)
