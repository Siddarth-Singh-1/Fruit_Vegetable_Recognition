# Fruit & Vegetable Classification

A deep learning image classifier that identifies 36 different types of fruits and vegetables using MobileNetV2 transfer learning.

## Quick Start

### Run the Classifier
```bash
cd local
python run_app.py
```
Opens web interface at http://127.0.0.1:7860

### Train a Model
```bash
cd local
python train_model.py
```

### Install Dependencies
```bash
cd local
pip install -r requirements.txt
```

## Project Structure

```
├── local/               # Local execution scripts
│   ├── run_app.py      # Web interface application
│   ├── train_model.py  # Training script
│   └── requirements.txt # Python dependencies
│
├── models/              # Pre-trained models
│   ├── FV.h5           # Main model (17 MB)
│   └── best_model.h5   # Best checkpoint
│
├── docs/               # Documentation
│   ├── README.md       # Detailed project overview
│   ├── START_HERE.md   # Getting started guide
│   └── ...             # Additional guides
│
├── notebooks/          # Jupyter notebooks
│   ├── Fruit_Vegetable_Recognition_Colab.ipynb  # Training notebook
│   └── Fruit_Vegetable_Classifier_UI.ipynb      # Inference notebook
│
└── dataset/            # Training and validation data
    ├── train/          # Training images
    └── validation/     # Validation images
```

## Features

- Classifies 36 different fruits and vegetables
- Uses MobileNetV2 transfer learning
- Interactive web interface (Gradio)
- Pre-trained model included
- Complete training pipeline
- Runs locally or in Google Colab

## Documentation

All documentation is in the `docs/` folder:

- [START_HERE.md](docs/START_HERE.md) - Quick start guide
- [README.md](docs/README.md) - Complete project overview
- [COMMANDS.md](docs/COMMANDS.md) - All commands
- [RUN_LOCALLY.md](docs/RUN_LOCALLY.md) - Installation guide
- [TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) - Training instructions
- [QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) - Quick reference

## System Requirements

- Python 3.8 or higher
- 4 GB RAM minimum (8 GB recommended)
- 2 GB disk space
- GPU recommended for training

## Supported Classes

**Fruits (15)**: Apple, Banana, Bell Pepper, Chilli Pepper, Grapes, Jalepeno, Kiwi, Lemon, Mango, Orange, Paprika, Pear, Pineapple, Pomegranate, Watermelon

**Vegetables (21)**: Beetroot, Cabbage, Capsicum, Carrot, Cauliflower, Corn, Cucumber, Eggplant, Garlic, Ginger, Lettuce, Onion, Peas, Potato, Radish, Soy Beans, Spinach, Sweet Corn, Sweet Potato, Tomato, Turnip

## Performance

- Validation accuracy: 85-95%
- Model size: ~17 MB
- Training time (GPU): 30-60 minutes
- Training time (CPU): 2-4 hours
- Inference: Real-time

## Dataset

Download from [Kaggle](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition)

Place in `dataset/` folder with structure:
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

## Team

- Siddart Singh (202310101360182)
- Adesh Srivastava (202310101360181)
- Vikash Maurya (202310101360194)

## License

MIT License - Free for educational and commercial use
