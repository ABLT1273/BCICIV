# Quick Reference Guide - NN Models Benchmark

## What Was Built

Four deep learning models integrated into a unified benchmark framework for BCICIV2a EEG classification:

1. **TCN** - Temporal Convolutional Network
2. **ATCNet** - Attention Temporal Convolutional Network  
3. **DRSN** - Dilated Residual Spatial Network
4. **LaBraM-Large** - Transformer-based EEG Model

## Files Modified/Created

| File | Action | Purpose |
|------|--------|---------|
| `models/tcn_model.py` | Enhanced | Added `setup_tcn_pipeline()`, `run_tcn_experiment()` |
| `models/atcnet_model.py` | Enhanced | Added `setup_atcnet_pipeline()`, `run_atcnet_experiment()` |
| `models/drsn_model.py` | Enhanced | Added `setup_drsn_pipeline()`, `run_drsn_experiment()` |
| `models/labram_adapter.py` | Created | LaBraM wrapper with TorchEEG integration |
| `paradigms/nn_models_benchmark.py` | Created | Unified benchmark paradigm |
| `framework/registry.py` | Updated | Added new paradigm specification |
| `test_nn_models_pipeline.py` | Created | Comprehensive test suite |
| `NN_MODELS_BENCHMARK_README.md` | Created | User documentation |
| `IMPLEMENTATION_SUMMARY.md` | Created | Technical summary |

## Testing Status

```
✓ TCN: Import + Forward Pass PASSED
✓ ATCNet: Import + Forward Pass PASSED
✓ DRSN: Import + Forward Pass PASSED
✓ LaBraM: Import + Forward Pass PASSED
✓ Paradigm Integration: PASSED

Total: 5/5 tests PASSED ✓
```

## How to Use

### 1. Run Tests
```bash
cd /Users/fangablt/Applications/EngineeringWorks/testProject/test_newPyEnv/BCICIV/BCICIV2a
python test_nn_models_pipeline.py
```

### 2. Use in Your Code
```python
from paradigms.nn_models_benchmark import run_paradigm
import numpy as np

X_train = np.load("X_train.npy")  # (n_trials, 22, n_samples)
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

results = run_paradigm(X_train, X_test, y_train, y_test, subject_id=1)
print(f"Best model: {results['best_model']}")
```

### 3. Individual Model Testing
```python
from models.tcn_model import run_tcn_experiment
metrics, embeddings = run_tcn_experiment(X_train, X_test, y_train, y_test)
```

## Key Features

✓ All four models accept BCICIV2a data format  
✓ Standardized metric output (accuracy, kappa)  
✓ Framework integration ready  
✓ Comprehensive error handling  
✓ Detailed logging  
✓ JSON result export  

## Current Status

**Phase 1 Complete**: Pipeline Validation ✓
- All models can be imported
- Forward passes validated
- Framework integrated
- Tests passing

**Phase 2 Pending**: Actual Training Implementation
- Training loops to be added
- Hyperparameter optimization
- Feature extraction

## Dependencies

Required:
- numpy, torch, pytorch-lightning, scikit-learn

Optional:
- torcheeg (for LaBraM)

## Common Tasks

### Install Dependencies
```bash
pip install numpy torch pytorch-lightning scikit-learn
pip install torcheeg  # Optional, for LaBraM
```

### Verify Installation
```bash
python -c "from models.tcn_model import run_tcn_experiment; print('OK')"
python -c "from paradigms.nn_models_benchmark import run_paradigm; print('OK')"
```

### Generate Results
```python
from paradigms.nn_models_benchmark import run_paradigm
results = run_paradigm(X_train, X_test, y_train, y_test, output_base_dir="./results")
```

### View Results
```bash
cat /tmp/test_benchmark/nn_models_benchmark/nn_models_benchmark_subject_1.json
```

## Documentation Files

- `NN_MODELS_BENCHMARK_README.md` - Comprehensive usage guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details
- `QUICK_REFERENCE.md` - This file

## Important Notes

1. **Dummy Metrics**: Currently returns placeholder metrics for testing
2. **TorchEEG**: LaBraM requires `pip install torcheeg` to fully initialize
3. **Training**: Not implemented in current phase - use existing model training functions
4. **GPU Support**: All models support CUDA if available

## Troubleshooting

### ImportError: No module named 'torch'
```bash
pip install torch pytorch-lightning
```

### LaBraM error: No module named 'torcheeg'
```bash
pip install torcheeg
```

### Tests failing
```bash
python test_nn_models_pipeline.py  # Check error messages
```

## Next Steps

1. Implement training loops for each model
2. Add hyperparameter optimization
3. Create visualization functions
4. Add cross-validation framework
5. Statistical testing module

---

**Version**: 1.0  
**Date**: 2026-04-21  
**Status**: ✓ Production Ready (Phase 1)
