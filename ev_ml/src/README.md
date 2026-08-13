# Source Code

This folder contains production-ready code for the EV ML project.

## Purpose

* Production training notebooks (`battery_energy_prediction_training.py`)
* Reusable Python modules
* Feature engineering functions
* Model training utilities
* Data preprocessing pipelines
* Evaluation metrics and reporting

## Usage

Modules in this folder can be imported in notebooks and jobs:

```python
from src.preprocessing import prepare_features
from src.models import train_xgboost_model
```

## Best Practices

* Write unit tests for all functions
* Keep functions focused and reusable
* Document with docstrings
* Use type hints