# Simple Command Guide

Easy-to-use commands for the Fruit & Vegetable Classification project.

## First Time Setup

### Step 1: Go to the local folder
```bash
cd local
```

### Step 2: Install everything
```bash
pip install -r requirements.txt
```

That's it! You're ready to go.

## Running the App

### Start the web interface
```bash
cd local
python run_app.py
```

Then open your browser to: http://127.0.0.1:7860

### Stop the app
Press `Ctrl + C` in the terminal

## Training a Model

### Start training
```bash
cd local
python train_model.py
```

Wait for training to complete (30-60 minutes with GPU, 2-4 hours with CPU)

## Check if Everything Works

### Check Python version
```bash
python --version
```
Should show Python 3.8 or higher

### Check if TensorFlow is installed
```bash
python -c "import tensorflow; print('TensorFlow is installed!')"
```

### Check if you have a GPU
```bash
python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

### See all installed packages
```bash
pip list
```

## Common Problems & Solutions

### Problem: "Module not found" error
**Solution:** Install dependencies again
```bash
cd local
pip install -r requirements.txt
```

### Problem: Port already in use
**Solution:** Close other programs using port 7860, or restart your computer

### Problem: Out of memory during training
**Solution:** Close other programs, or reduce batch size in train_model.py

## If Something Breaks

### Reinstall everything
```bash
cd local
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Start fresh
```bash
cd local
pip install tensorflow numpy pillow matplotlib scikit-learn gradio
```

## Test Your Setup

### Test if model file exists
```bash
cd models
dir FV.h5
```
Should show the file (about 17 MB)

### Test if Gradio works
```bash
python -c "import gradio; print('Gradio is ready!')"
```

## Complete Workflows

### First time using the project
```bash
# 1. Go to local folder
cd local

# 2. Install everything
pip install -r requirements.txt

# 3. Run the app
python run_app.py

# 4. Open browser to http://127.0.0.1:7860
```

### Training your own model
```bash
# 1. Make sure dataset is ready
# (Should have dataset/train/ and dataset/validation/ folders)

# 2. Go to local folder
cd local

# 3. Start training
python train_model.py

# 4. Wait for it to finish
# 5. Run the app with your new model
python run_app.py
```

## Customization

### Change port (if 7860 is busy)
Open `local/run_app.py` and find this line:
```python
server_port=7860
```
Change `7860` to any other number like `8080`

### Make training faster (less accurate)
Open `local/train_model.py` and change:
```python
EPOCHS = 10  # Change from 20 to 10
```

### Make training use less memory
Open `local/train_model.py` and change:
```python
BATCH_SIZE = 8  # Change from 32 to 8
```

## Emergency: Nothing Works!

### Nuclear option - reinstall everything
```bash
cd local
pip uninstall tensorflow numpy pillow matplotlib scikit-learn gradio -y
pip install -r requirements.txt
```

### Still not working?
1. Restart your computer
2. Try the commands again
3. Check if you have Python 3.8 or higher: `python --version`

## Tips

### During training, you'll see:
- Progress bar showing epochs
- Accuracy numbers (higher is better)
- Loss numbers (lower is better)
- Files created: `training_history.png`, `sample_images.png`, `test_predictions.png`

### Monitor your computer
Press `Ctrl + Shift + Esc` to open Task Manager and watch:
- CPU usage
- RAM usage
- GPU usage (if you have one)

## Quick Reference

| What you want to do | Command |
|---------------------|----------|
| Install everything | `cd local` then `pip install -r requirements.txt` |
| Run the app | `cd local` then `python run_app.py` |
| Train a model | `cd local` then `python train_model.py` |
| Stop the app | Press `Ctrl + C` |
| Check Python version | `python --version` |

---

**Remember**: Always run commands from the `local/` folder!
