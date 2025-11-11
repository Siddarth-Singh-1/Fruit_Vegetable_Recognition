# Model Training Guide

This guide covers the complete process for training a custom fruit and vegetable classification model.

## Prerequisites

### Hardware Requirements
- 8 GB RAM minimum (16 GB recommended)
- 5 GB available disk space
- NVIDIA GPU with CUDA support (optional, significantly improves training speed)

### Software Requirements
- Python 3.8 or higher
- CUDA toolkit (if using GPU)
- Dataset from Kaggle

## Dataset Preparation

### Obtaining the Dataset

Download the dataset from Kaggle:
- URL: https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition
- Size: Approximately 1 GB
- Contains: 36 classes of fruits and vegetables

### Dataset Organization

Extract and organize the dataset in the following structure:

```
project_root/
├── dataset/
│   ├── train/
│   │   ├── apple/
│   │   ├── banana/
│   │   ├── beetroot/
│   │   └── ... (36 classes total)
│   └── validation/
│       ├── apple/
│       ├── banana/
│       ├── beetroot/
│       └── ... (36 classes total)
├── train_model.py
└── ...
```

The `dataset` directory must be in the same location as `train_model.py`.

## Training Process

### Basic Training

Execute the training script:
```bash
python train_model.py
```

### Training Configuration

Modify parameters in `train_model.py` as needed:

```python
# Image and batch configuration
IMG_SIZE = 224          # Input image dimensions
BATCH_SIZE = 32         # Batch size (reduce if memory limited)
EPOCHS = 20             # Training epochs
LEARNING_RATE = 0.0001  # Optimizer learning rate
```

### Hardware-Specific Adjustments

**Limited Memory Systems:**
```python
BATCH_SIZE = 8   # Reduces memory usage
EPOCHS = 10      # Shorter training time
```

**High-Performance Systems:**
```python
BATCH_SIZE = 64  # Faster training with more memory
EPOCHS = 30      # More thorough training
```

## Training Pipeline

The training script executes the following steps:

1. **Dataset Validation** - Verifies dataset structure and class counts
2. **Data Augmentation** - Applies transformations to training images
3. **Model Construction** - Builds MobileNetV2-based architecture
4. **Training Loop** - Trains with callbacks for optimization
5. **Evaluation** - Tests model performance on validation set
6. **Visualization** - Generates training curves and sample predictions

## Training Callbacks

The script implements three callbacks:

- **Early Stopping**: Halts training if validation loss plateaus
- **Learning Rate Reduction**: Decreases learning rate when progress stalls
- **Model Checkpointing**: Saves best-performing model during training

## Output Files

Training generates the following files:

| File | Description | Size |
|------|-------------|------|
| `FV.h5` | Final trained model | ~17 MB |
| `best_model.h5` | Best checkpoint | ~17 MB |
| `training_history.png` | Accuracy/loss curves | ~200 KB |
| `sample_images.png` | Training data samples | ~500 KB |
| `test_predictions.png` | Validation predictions | ~500 KB |

## Expected Performance

### Training Metrics
- Training accuracy: 90-95%
- Validation accuracy: 85-92%
- Training time (GPU): 30-60 minutes
- Training time (CPU): 2-4 hours

### Model Specifications
- Architecture: MobileNetV2 with custom classification head
- Input shape: 224×224×3
- Parameters: Approximately 3.5 million
- Output classes: 36

## Troubleshooting

### Out of Memory Errors

Reduce batch size in `train_model.py`:
```python
BATCH_SIZE = 8  # or even 4 for very limited systems
```

### Dataset Not Found

Verify the dataset directory structure matches the required format. Ensure all 36 class folders exist in both `train/` and `validation/` directories.

### Slow Training Performance

Without GPU acceleration, training is significantly slower. Consider:
- Reducing epochs: `EPOCHS = 10`
- Reducing batch size: `BATCH_SIZE = 16`
- Using cloud platforms with GPU support (Google Colab)

### CUDA/GPU Not Detected

Verify CUDA installation:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

Ensure TensorFlow version matches CUDA version requirements.

## Advanced Options

### Resuming Training

To continue from a checkpoint, modify `train_model.py`:
```python
# After model construction
model = load_model('best_model.h5')
```

### Fine-Tuning

To unfreeze base model layers for fine-tuning:
```python
base_model.trainable = True
model.compile(optimizer=Adam(learning_rate=0.00001), ...)
```

### Custom Dataset

For different classes:
1. Organize images in the same directory structure
2. Update the `labels` dictionary in `train_model.py`
3. Modify `NUM_CLASSES` to match your dataset
4. Execute training script

## Validation

After training, validate the model:
```bash
python run_app.py
```

Test with various images to ensure proper classification across all classes.

## Performance Optimization

1. Enable GPU acceleration when available
2. Use appropriate batch sizes for your hardware
3. Monitor training curves for overfitting
4. Adjust learning rate if convergence is slow
5. Implement data augmentation for better generalization

## Additional Resources

- Complete command reference: `COMMANDS.md`
- Local setup instructions: `RUN_LOCALLY.md`
- Project overview: `README.md`
