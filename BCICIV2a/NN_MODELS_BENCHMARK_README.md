# NN Models Benchmark for BCICIV2a

## Overview

This benchmark provides a unified framework for testing four state-of-the-art deep learning models for EEG-based motor imagery classification on the BCICIV2a dataset:

1. **TCN** (Temporal Convolutional Network)
   - Pure temporal convolution-based approach
   - Status: Pipeline validated ✓
   - Implementation: Ready for training integration

2. **ATCNet** (Attention Temporal Convolutional Network)
   - Combines temporal convolutions with attention mechanisms
   - Status: Pipeline validated ✓
   - Implementation: Ready for training integration

3. **DRSN** (Dilated Residual Spatial Network)
   - Uses dilated convolutions for wider receptive fields
   - Spatial filtering with residual connections
   - Status: Pipeline validated ✓
   - Implementation: Ready for training integration

4. **LaBraM-Large** (Transformer-based EEG model)
   - Official model from TorchEEG
   - Patch-based transformer architecture
   - Status: Forward pass validated ✓
   - Note: Requires TorchEEG installation

## Project Structure

```
models/
├── tcn.py                  # TCN adapter
├── atcnet.py              # ATCNet adapter
├── drsn.py                # DRSN adapter
└── labram_adapter.py      # LaBraM wrapper

paradigms/
└── nn_models_benchmark.py # Unified benchmark paradigm

framework/
└── registry.py            # Updated with new paradigm spec

test_nn_models_pipeline.py # Validation test script
```

## Installation

### Basic Requirements
```bash
pip install numpy torch pytorch-lightning scikit-learn
```

### For LaBraM Support
```bash
pip install torcheeg
```

## Usage

### 1. Run Pipeline Validation Test

Test that all models can be imported and forward passes work:

```bash
python test_nn_models_pipeline.py
```

Expected output:
- ✓ TCN import and forward pass validated
- ✓ ATCNet import and forward pass validated  
- ✓ DRSN import and forward pass validated
- ✓ LaBraM import and forward pass validated (or warning if TorchEEG not installed)
- ✓ Paradigm integration successful

### 2. Use Paradigm in Your Code

```python
from paradigms.nn_models_benchmark import run_paradigm
import numpy as np

# Load your BCICIV2a data
X_train = np.load("X_train.npy")  # (n_trials, 22_channels, n_samples)
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

# Run benchmark
results = run_paradigm(
    X_train, X_test, y_train, y_test,
    subject_id=1,
    output_base_dir="./results"
)

# Access results
print(f"Best model: {results['best_model']}")
print(f"Mean accuracy: {results['mean_accuracy']:.4f}")
```

### 3. Individual Model Testing

Test a single model:

```python
from models.tcn import run_tcn_experiment
import numpy as np

X_train = np.random.randn(20, 22, 1500)
X_test = np.random.randn(10, 22, 1500)
y_train = np.random.randint(0, 4, 20)
y_test = np.random.randint(0, 4, 10)

metrics, embeddings = run_tcn_experiment(X_train, X_test, y_train, y_test)

print(f"TCN Accuracy: {metrics['accuracy']:.4f}")
print(f"TCN Kappa: {metrics['kappa']:.4f}")
```

## Current Status

### ✓ Completed
- [x] TCN adapter with configurable architecture
- [x] ATCNet adapter with attention mechanisms
- [x] DRSN adapter with dilated convolutions
- [x] LaBraM wrapper for TorchEEG integration
- [x] Forward pass validation for all models
- [x] Unified paradigm framework
- [x] Result aggregation and comparison
- [x] Pipeline validation test suite

### ⏳ Future Work
- [ ] Implement actual training loops for all models
- [ ] Add hyperparameter optimization
- [ ] Feature extraction from intermediate layers
- [ ] 3D visualization of embeddings
- [ ] Cross-validation and statistical testing
- [ ] Result export to various formats (CSV, JSON, plots)

## Model Specifications

### TCN
```
- Input: (batch, 22_channels, time_steps)
- Output: (batch, 4_classes)
- Architecture:
  - Temporal conv blocks with dilation = [1, 2, 4, 8]
  - Kernel size: 5
  - Channels: [32, 64, 128, 256]
  - Batch normalization and dropout
```

### ATCNet
```
- Input: (batch, 22_channels, time_steps)
- Output: (batch, 4_classes)
- Architecture:
  - Temporal conv branch (kernel 5, 10, 15)
  - Spatial conv branch (pointwise)
  - Multi-head attention
  - Feature fusion + classification head
```

### DRSN
```
- Input: (batch, 22_channels, time_steps)
- Output: (batch, 4_classes)
- Architecture:
  - Initial spatial conv (22 → 32)
  - Dilated residual blocks (dilation = 1, 2, 4, 8)
  - Global average pooling
  - Dense classification head
```

### LaBraM-Large
```
- Input: (batch, 22_channels, 1600_samples)
- Output: (batch, 4_classes)
- Preprocesses as patches:
  - Patch size: 200 samples
  - Number of patches: 8 (1600 / 200)
- Architecture:
  - Temporal-spatial patch embedding
  - 12-layer transformer with 10 attention heads
  - Embedding dimension: 200
- Note: Requires TorchEEG: pip install torcheeg
```

## Input Data Format

All models expect:
- **Input shape**: `(batch_size, 22_channels, time_steps)`
- **Channel order**: Standard BCICIV2a electrode order
- **Sampling rate**: 250 Hz (implicit)
- **Data type**: float32

Typical BCICIV2a data:
- 22 EEG channels
- Time window: 4 seconds (1000 samples at 250 Hz) to 8 seconds
- 4 motor imagery classes: left hand, right hand, feet, tongue

## Output Format

Each model returns:
```python
{
    "accuracy": float,      # 0.0 to 1.0
    "kappa": float,         # -1.0 to 1.0 (Cohen's kappa)
    "embeddings": ndarray   # (n_test, embedding_dim)
}
```

## Troubleshooting

### ImportError: No module named 'torch'
```bash
pip install torch pytorch-lightning
```

### LaBraM ImportError: No module named 'torcheeg'
```bash
pip install torcheeg
# See: https://torcheeg.readthedocs.io/
```

### CUDA Out of Memory
- Reduce batch size in model configuration
- Use CPU device explicitly (set `device='cpu'`)

### Slow Forward Pass
- Check GPU availability: `torch.cuda.is_available()`
- Verify model is on GPU: `model.to(device)`
- Profile with `torch.profiler`

## References

- BCICIV2a Dataset: http://www.bbci.de/competition/iv/
- TCN: Bai et al., 2018. "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling"
- ATCNet: Altaheri et al., 2021. "Deep Learning Techniques for Classification of Electroencephalogram (EEG) Motor Imagery"
- DRSN: Wang et al., 2022
- LaBraM: TorchEEG official models
- TorchEEG: https://torcheeg.readthedocs.io/

## License

This benchmark framework follows the same license as the BCICIV project.
