# MNIST Classifier - Clean Solution

A well-organized MNIST digit classification project.

## Structure

```
_solution/
├── train.py           # Main training script
├── config/
│   └── config.yaml    # Configuration
├── src/
│   ├── model.py       # CNN model
│   ├── dataset.py     # Data loading
│   ├── trainer.py     # Training logic
│   ├── metrics.py     # Evaluation metrics
│   └── utils.py       # Utilities
├── tests/
│   ├── test_model.py
│   └── test_utils.py
└── requirements.txt
```

## Key Improvements from Original

1. **Removed duplicate files** - Consolidated 16 train*.py files into 1
2. **Removed dead code** - Deleted unused functions and classes
3. **Extracted magic numbers** - Moved to config.yaml
4. **Single responsibility** - Each module has one clear purpose
5. **Type hints** - Added throughout for clarity
6. **Tests** - Proper pytest structure

## Usage

```bash
python train.py --epochs 10 --batch-size 64 --lr 0.001
```

## Testing

```bash
pytest tests/
```
