# Quick Reference

## Essential Commands

### Run Inference
```bash
python run_app.py
```

### Train Model
```bash
python train_model.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Configuration

### Memory Optimization
Edit `train_model.py`:
```python
BATCH_SIZE = 8  # Reduce for lower memory usage
```

### Training Speed vs Accuracy
Edit `train_model.py`:
```python
EPOCHS = 10  # Fewer epochs = faster training
```

### Network Port
Edit `run_app.py`:
```python
server_port=7861  # Change if port 7860 is in use
```

## Common Tasks

### Using the Classifier
```bash
python run_app.py
```
Navigate to http://127.0.0.1:7860 and upload images.

### Training on Custom Dataset
1. Organize images in `dataset/train/` and `dataset/validation/`
2. Update class labels in `train_model.py` if needed
3. Run `python train_model.py`

### Check GPU Availability
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Resume Training
Edit `train_model.py` after model initialization:
```python
model = load_model('best_model.h5')
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `BATCH_SIZE` in `train_model.py` |
| Dataset not found | Verify `dataset/train/` and `dataset/validation/` exist |
| Model file missing | Ensure `FV.h5` is in project directory |
| Port conflict | Change `server_port` in `run_app.py` |
| Slow training | Use GPU or reduce `EPOCHS` |
| Import errors | Run `pip install -r requirements.txt` |

## Performance Metrics

| Metric | Expected Value |
|--------|---------------|
| Training accuracy | 90-95% |
| Validation accuracy | 85-92% |
| Model size | ~17 MB |
| Number of classes | 36 |
| Training time (GPU) | 30-60 minutes |
| Training time (CPU) | 2-4 hours |

## Dataset Information

**Source**: https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition

**Required Structure**:
```
dataset/
├── train/
│   ├── apple/
│   ├── banana/
│   └── ... (36 classes)
└── validation/
    ├── apple/
    ├── banana/
    └── ... (36 classes)
```

## Best Practices

1. Use GPU acceleration for training when available
2. Test with pre-trained model before custom training
3. Monitor training curves in `training_history.png`
4. Maintain proper dataset organization
5. Verify dependencies are correctly installed

## Documentation Links

- Setup instructions: `RUN_LOCALLY.md`
- Training details: `TRAINING_GUIDE.md`
- Project overview: `README.md`
- Complete commands: `COMMANDS.md`
