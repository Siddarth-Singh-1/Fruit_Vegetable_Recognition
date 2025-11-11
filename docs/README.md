# Fruit & Vegetable Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

A deep learning image classifier that identifies 36 different types of fruits and vegetables. Built with MobileNetV2 transfer learning for efficient and accurate predictions.

**Team Members**: 
- Siddart Singh (202310101360182)
- Adesh Srivastava (202310101360181)
- Vikash Maurya (202310101360194)

## Features

- Classifies 36 different fruits and vegetables
- Uses MobileNetV2 transfer learning for high accuracy
- Runs locally on your PC or in Google Colab
- Interactive web interface powered by Gradio
- Pre-trained model included for immediate use
- Complete training pipeline with data augmentation

## Quick Start

### Local PC Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the classifier:**
```bash
python run_app.py
```
This launches a web interface at http://127.0.0.1:7860 where you can upload images for classification.

**Train your own model:**
```bash
python train_model.py
```
Note: Training requires the dataset to be placed in `dataset/train/` and `dataset/validation/` folders.

For detailed commands and troubleshooting, see [COMMANDS.md](COMMANDS.md).

### Google Colab Setup

**For training:**
1. Open `Fruit_Vegetable_Recognition_Colab.ipynb` in Google Colab
2. Enable GPU in Runtime settings for faster training
3. Upload your dataset or connect to Google Drive
4. Run all cells to train the model
5. Download the trained model file when complete

**For inference only:**
1. Open `Fruit_Vegetable_Classifier_UI.ipynb` in Google Colab
2. Upload the pre-trained `FV.h5` model
3. Run the cells to launch the web interface
4. Start classifying images immediately

## Project Structure

```
├── run_app.py                                # Local UI application
├── train_model.py                            # Local training script
├── FV.h5                                     # Pre-trained model (17 MB)
├── COMMANDS.md                               # All commands reference
├── requirements.txt                          # Python dependencies
├── README.md                                 # Main documentation
├── Fruit_Vegetable_Recognition_Colab.ipynb  # Colab training notebook
├── Fruit_Vegetable_Classifier_UI.ipynb      # Colab UI notebook
└── dataset/                                  # Dataset folder
    ├── train/                                # Training images
    └── validation/                           # Validation images
```

## Implementation Details

### Training Pipeline
The training notebook (`Fruit_Vegetable_Recognition_Colab.ipynb`) includes:
- Automated data loading and preprocessing
- Image augmentation techniques for better generalization
- MobileNetV2 base model with custom classification layers
- Training callbacks: early stopping, learning rate scheduling, model checkpointing
- Performance visualization and evaluation metrics
- Gradio web interface for testing predictions

### Inference Application
The UI notebook (`Fruit_Vegetable_Classifier_UI.ipynb`) provides:
- Simple interface for loading pre-trained models
- Real-time image classification
- Confidence scores for top predictions
- Category identification (fruit vs vegetable)

## Model Architecture

- **Base Model**: MobileNetV2 pre-trained on ImageNet
- **Input Shape**: 224×224×3 RGB images
- **Output**: 36 class probabilities (softmax activation)
- **Training Strategy**: Transfer learning with frozen base layers
- **Optimization**: Adam optimizer with adaptive learning rate
- **Performance**: Achieves 85-95% validation accuracy

## Supported Classes

The model recognizes 36 classes across fruits and vegetables:

**Fruits (15)**: Apple, Banana, Bell Pepper, Chilli Pepper, Grapes, Jalepeno, Kiwi, Lemon, Mango, Orange, Paprika, Pear, Pineapple, Pomegranate, Watermelon

**Vegetables (21)**: Beetroot, Cabbage, Capsicum, Carrot, Cauliflower, Corn, Cucumber, Eggplant, Garlic, Ginger, Lettuce, Onion, Peas, Potato, Radish, Soy Beans, Spinach, Sweet Corn, Sweet Potato, Tomato, Turnip

## Dataset

The training dataset is available on [Kaggle](https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition).

Dataset structure:
```
dataset/
├── train/          # Training images organized by class
└── validation/     # Validation images organized by class
```

Each class folder contains images of the corresponding fruit or vegetable.

## Requirements

Python 3.8 or higher with the following packages:
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Pillow >= 9.0.0
- Matplotlib >= 3.5.0
- Scikit-learn >= 1.0.0
- Gradio >= 3.50.0

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Performance Notes

- GPU acceleration recommended for training (10-20x faster than CPU)
- Expected training time: 30-60 minutes with GPU, 2-4 hours with CPU
- Model file size: approximately 17 MB
- Inference speed: Real-time predictions on standard hardware

## License

MIT License - Free to use for educational and commercial purposes.
