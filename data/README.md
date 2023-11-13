# Data

Download datasets with the `get_dataset` command.

```bash
Usage: get_dataset DATASET_NAME
Options:
        jets: Classifying simulated jets of proton-proton colisions.
        econ: Data generated with the Level 1 Trigger Primitives simulation.
        ic: Image classification on CIFAR10.
        ad: Unsupervised detection of anomalous sounds.
```

## Note

The `ECON-T` dataset must be processed. Use the following:

```bash
training/econ/scripts/preprocess_data.sh
```
