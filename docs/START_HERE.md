# Getting Started

## Basic Commands

### Run the Web Interface
```bash
python run_app.py
```
Launches the classification interface at http://127.0.0.1:7860

### Train a New Model
```bash
python train_model.py
```
Requires dataset in `dataset/train/` and `dataset/validation/` folders

## Initial Setup

### Install Dependencies
```bash
pip install -r requirements.txt --no-cache-dir
```

Alternatively, install packages individually:
```bash
pip install tensorflow numpy pillow matplotlib scikit-learn gradio seaborn --no-cache-dir
```

## Documentation

- [COMMANDS.md](COMMANDS.md) - Complete command reference
- [README.md](README.md) - Project overview and features
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training instructions
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - Quick reference guide

## Project Components

- `run_app.py` - Web interface application
- `train_model.py` - Model training script
- `FV.h5` - Pre-trained model file
- `dataset/` - Dataset directory structure

## Recommended Workflow

1. Install dependencies using the command above
2. Test the pre-trained model by running `python run_app.py`
3. Upload sample images to verify functionality
4. For custom training, prepare your dataset and run `python train_model.py`

For detailed instructions and troubleshooting, refer to [COMMANDS.md](COMMANDS.md).
