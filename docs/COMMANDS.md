# Command Reference

Complete reference for all commands used in the Fruit & Vegetable Classification project.

## Installation Commands

### Install All Dependencies
```bash
pip install tensorflow>=2.10.0 numpy>=1.21.0 Pillow>=9.0.0 matplotlib>=3.5.0 scikit-learn>=1.0.0 gradio>=3.50.0 seaborn --no-cache-dir
```

### Install from Requirements File
```bash
pip install -r requirements.txt
```

### Install Without Cache (Memory-Constrained Systems)
```bash
pip install -r requirements.txt --no-cache-dir
```

### Upgrade Pip
```bash
python -m pip install --upgrade pip
```

## Execution Commands

### Run Web Interface
```bash
python run_app.py
```
Starts the classification interface at http://127.0.0.1:7860

### Train Model
```bash
python train_model.py
```
Initiates training process using dataset in `dataset/` directory

## Verification Commands

### Check Python Version
```bash
python --version
```

### Verify TensorFlow Installation
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Check GPU Availability
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### List Installed Packages
```bash
pip list
```

### Verify Model File
```bash
dir FV.h5
```
(Windows) or `ls -lh FV.h5` (Linux/Mac)

### Check Dataset Structure
```bash
dir dataset\train
dir dataset\validation
```

## Maintenance Commands

### Clear Pip Cache
```bash
pip cache purge
```

### Remove Virtual Environment
```bash
rmdir /s /q venv
```
(Windows) or `rm -rf venv` (Linux/Mac)

### Clean Training Outputs
```bash
del best_model.h5 training_history.png sample_images.png test_predictions.png
```

## Troubleshooting Commands

### Reinstall TensorFlow
```bash
pip uninstall tensorflow
pip install tensorflow>=2.10.0 --no-cache-dir
```

### Force Reinstall All Dependencies
```bash
pip install --upgrade --force-reinstall tensorflow numpy pillow matplotlib scikit-learn gradio
```

### Fix Import Errors
```bash
pip uninstall -r requirements.txt -y
pip install -r requirements.txt --no-cache-dir
```

## Testing Commands

### Test Model Loading
```bash
python -c "from tensorflow.keras.models import load_model; model = load_model('FV.h5'); print('Model loaded successfully')"
```

### Test Gradio Installation
```bash
python -c "import gradio as gr; print(f'Gradio version: {gr.__version__}')"
```

### Test Image Processing
```bash
python -c "from PIL import Image; import numpy as np; print('PIL and NumPy functional')"
```

## Network Commands

### Check Port Availability
```bash
netstat -ano | findstr :7860
```

### Terminate Process on Port
```bash
# Find process ID
netstat -ano | findstr :7860

# Kill process (replace <PID> with actual process ID)
taskkill /PID <PID> /F
```

## Dataset Commands

### Count Training Images (Windows)
```bash
for /d %d in (dataset\train\*) do @echo %d && dir "%d\*.*" /b | find /c /v ""
```

### Verify Dataset Integrity
```bash
python -c "import os; print('Train classes:', len(os.listdir('dataset/train'))); print('Val classes:', len(os.listdir('dataset/validation')))"
```

## Common Workflows

### Initial Setup Workflow
```bash
# Verify Python
python --version

# Install dependencies
pip install -r requirements.txt --no-cache-dir

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Run application
python run_app.py
```

### Training Workflow
```bash
# Verify dataset
dir dataset\train
dir dataset\validation

# Check GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Start training
python train_model.py

# Use trained model
python run_app.py
```

### Update Workflow
```bash
# Update pip
python -m pip install --upgrade pip

# Update packages
pip install --upgrade tensorflow numpy pillow matplotlib scikit-learn gradio

# Verify updates
pip list
```

## Configuration Modifications

### Change Server Port
Edit `run_app.py`:
```python
demo.launch(
    server_name="127.0.0.1",
    server_port=7861,  # Change port number
    share=False,
    inbrowser=True
)
```

### Adjust Training Parameters
Edit `train_model.py`:
```python
BATCH_SIZE = 16    # Reduce for memory constraints
EPOCHS = 10        # Reduce for faster training
LEARNING_RATE = 0.0001
```

### Resume Training from Checkpoint
Edit `train_model.py`, add after model construction:
```python
model = load_model('best_model.h5')
```

## Emergency Recovery

### Complete Reset
```bash
# Remove virtual environment
rmdir /s /q venv

# Clear cache
pip cache purge

# Fresh installation
pip install tensorflow numpy pillow matplotlib scikit-learn gradio seaborn --no-cache-dir

# Test
python run_app.py
```

### Port Conflict Resolution
```bash
# Identify process
netstat -ano | findstr :7860

# Terminate process
taskkill /PID <PID> /F

# Restart application
python run_app.py
```

### Model Verification
```bash
# Check file existence and size
dir FV.h5

# Verify model loads correctly
python -c "from tensorflow.keras.models import load_model; load_model('FV.h5')"
```

## Performance Monitoring

### Monitor Training Progress
Training metrics are displayed in real-time during execution. Generated files include:
- `training_history.png` - Training and validation curves
- `sample_images.png` - Sample training data
- `test_predictions.png` - Model predictions on test data

### Check System Resources
```bash
# Windows Task Manager: Ctrl+Shift+Esc
# Monitor CPU, RAM, and GPU usage during training
```

## Additional Notes

- All commands assume execution from the project root directory
- Windows commands are shown; Linux/Mac equivalents may differ slightly
- GPU commands require CUDA-compatible hardware and drivers
- Training commands require properly structured dataset
- Port 7860 is default; modify if conflicts occur
