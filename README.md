# BinaryStringClassifier 🔣

An exploration into SMILES detection using a LSTM neural network.

Further discussion can be found in the associated [blog post](https://mattravenhall.github.io/2022/04/03/BinaryStringClassifier.html)

## Quick Start
```bash
# Install
pip install -r requirements.txt
python3 setup.py install

# View help
BSC-Data -h
BSC-Model -h
```

## Functionality
- `BSC-Data`: Entrypoint for dataset creation and curation.
	- `create`: Create randomly generated datasets within given parameters.
	- `combine`: Combine multiple components datasets into one ready for training.
	- `evaluate`: Generate summary information for given a dataset.
	- `split`: Perform train/test split for a given dataset.
- `BSC-Model`: Entrypoint for model training & evaluation.
	- `train`: Train a BinaryStringClassifier model.
	- `evaluate`: Evaluate a trained BinaryStringClassifier model.
	- `predict`: Get SMILES probabilities with a trained BinaryStringClassifier model

## Example Plots
|     |     |
| --- | --- |
| ![Accuracy by Epoch](./examples/accuracy_by_epoch.png) | ![Loss by Epoch](./examples/loss_by_epoch.png) |
| ![Confusion Matrix](./examples/confusion_matrix.png) | ![ROC Curve](./examples/roc_curve.png) |
