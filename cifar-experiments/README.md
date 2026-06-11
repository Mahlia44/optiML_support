# CIFAR-10 experiments

This folder contains preliminary CIFAR-10 experiments used as a realistic extension attempt of the synthetic support-identification setup. These results are not central to the report conclusions; they support the limitation that realistic feature settings are harder to interpret because relevant and irrelevant coordinates are not known.

## Files 

- `train_cifar_resnet_head.py`: trains an MLP head on frozen ResNet18 CIFAR-10 embeddings.
- `notebooks/analyze_cifar_resnet_head.ipynb`: main analysis notebook for the CIFAR runs.
- `notebooks/optiml_results_analysis.ipynb`: exploratory/secondary analysis notebook.
- `cifar-results/`: saved outputs from the CIFAR runs.
- `cifar10_models/`: model definitions used for CIFAR backbones or checkpoint compatibility.
