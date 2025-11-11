# Project Summary

## Overview

This is a deep learning-based image classification system that identifies 36 different types of fruits and vegetables using MobileNetV2 transfer learning architecture. The project includes both training and inference capabilities, with support for local execution and cloud-based deployment.

## Project Status

**Current Version**: Production-ready  
**Model**: Pre-trained MobileNetV2 (FV.h5 - 17 MB)  
**Accuracy**: 85-95% validation accuracy  
**Classes**: 36 (15 fruits, 21 vegetables)

## Core Components

### Python Scripts
- `run_app.py` - Web-based inference application using Gradio
- `train_model.py` - Complete training pipeline with data augmentation

### Pre-trained Model
- `FV.h5` - Main model file (17 MB)
- `best_model.h5` - Best checkpoint from training

### Documentation
- `README.md` - Project overview and quick start
- `START_HERE.md` - Getting started guide
- `COMMANDS.md` - Complete command reference
- `RUN_LOCALLY.md` - Local installation instructions
- `TRAINING_GUIDE.md` - Model training documentation
- `QUICK_REFERENCE.md` - Quick reference guide

### Jupyter Notebooks
- `Fruit_Vegetable_Recognition_Colab.ipynb` - Training notebook for Google Colab
- `Fruit_Vegetable_Classifier_UI.ipynb` - Inference notebook for Google Colab

### Configuration
- `requirements.txt` - Python dependencies
- `dataset/` - Dataset directory structure (train/validation splits)

## Technical Specifications

### Model Architecture
- **Base**: MobileNetV2 pre-trained on ImageNet
- **Input**: 224×224×3 RGB images
- **Output**: 36-class softmax predictions
- **Parameters**: ~3.5 million
- **Optimization**: Adam optimizer with adaptive learning rate

### Training Configuration
- **Batch Size**: 32 (adjustable)
- **Epochs**: 20 (adjustable)
- **Learning Rate**: 0.0001
- **Data Augmentation**: Rotation, shift, zoom, flip
- **Callbacks**: Early stopping, LR reduction, checkpointing

### Performance Metrics
- Training accuracy: 90-95%
- Validation accuracy: 85-92%
- Training time (GPU): 30-60 minutes
- Training time (CPU): 2-4 hours
- Inference: Real-time on standard hardware

## Supported Classes

### Fruits (15)
Apple, Banana, Bell Pepper, Chilli Pepper, Grapes, Jalepeno, Kiwi, Lemon, Mango, Orange, Paprika, Pear, Pineapple, Pomegranate, Watermelon

### Vegetables (21)
Beetroot, Cabbage, Capsicum, Carrot, Cauliflower, Corn, Cucumber, Eggplant, Garlic, Ginger, Lettuce, Onion, Peas, Potato, Radish, Soy Beans, Spinach, Sweet Corn, Sweet Potato, Tomato, Turnip

## Usage Instructions

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run inference
python run_app.py

# Train model
python train_model.py
```

### Web Interface
The application launches a Gradio interface at http://127.0.0.1:7860 where users can upload images for classification. Results include:
- Primary prediction with confidence score
- Category classification (fruit/vegetable)
- Top-3 predictions with probabilities

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4 GB RAM
- 2 GB disk space
- CPU-based execution

### Recommended Requirements
- Python 3.8+
- 8 GB RAM
- 5 GB disk space
- NVIDIA GPU with CUDA support
- 16 GB RAM for training

## Dataset Information

**Source**: Kaggle - Fruit and Vegetable Image Recognition  
**URL**: https://www.kaggle.com/kritikseth/fruit-and-vegetable-image-recognition  
**Size**: ~1 GB  
**Structure**: Organized by class in train/validation splits

## Dependencies

### Core Libraries
- TensorFlow >= 2.10.0
- NumPy >= 1.21.0
- Pillow >= 9.0.0
- Matplotlib >= 3.5.0
- Scikit-learn >= 1.0.0
- Gradio >= 3.50.0
- Seaborn

## Training Process

The training pipeline includes:
1. Dataset validation and verification
2. Data augmentation (rotation, shift, zoom, flip)
3. Model construction with MobileNetV2 base
4. Training with callbacks (early stopping, LR scheduling)
5. Performance evaluation on validation set
6. Visualization generation (curves, predictions)

### Training Outputs
- `FV.h5` - Final trained model
- `best_model.h5` - Best checkpoint
- `training_history.png` - Training curves
- `sample_images.png` - Sample data
- `test_predictions.png` - Test predictions

## Deployment Options

### Local Deployment
Run directly on local machine with Python environment. Suitable for development and testing.

### Cloud Deployment
Use Google Colab notebooks for training with free GPU access. Suitable for users without local GPU resources.

## Project Structure

```
project_root/
├── run_app.py                    # Inference application
├── train_model.py                # Training script
├── FV.h5                         # Pre-trained model
├── best_model.h5                 # Best checkpoint
├── requirements.txt              # Dependencies
├── README.md                     # Main documentation
├── START_HERE.md                 # Getting started
├── COMMANDS.md                   # Command reference
├── RUN_LOCALLY.md                # Installation guide
├── TRAINING_GUIDE.md             # Training documentation
├── QUICK_REFERENCE.md            # Quick reference
├── PROJECT_SUMMARY.md            # This file
├── dataset/                      # Dataset directory
│   ├── train/                    # Training images
│   └── validation/               # Validation images
├── Fruit_Vegetable_Recognition_Colab.ipynb
├── Fruit_Vegetable_Classifier_UI.ipynb
└── venv/                         # Virtual environment (optional)
```

## Key Features

1. **Transfer Learning**: Leverages pre-trained MobileNetV2 for efficient training
2. **Data Augmentation**: Improves generalization through image transformations
3. **Interactive Interface**: User-friendly web interface for classification
4. **Flexible Training**: Configurable parameters for different hardware
5. **Comprehensive Documentation**: Detailed guides for all use cases
6. **Dual Deployment**: Supports both local and cloud execution
7. **Real-time Inference**: Fast predictions on standard hardware
8. **Model Checkpointing**: Saves best model during training

## Troubleshooting

Common issues and solutions are documented in:
- `COMMANDS.md` - Command-line solutions
- `RUN_LOCALLY.md` - Installation issues
- `TRAINING_GUIDE.md` - Training problems

## Future Enhancements

Potential improvements:
- Additional class support
- Model quantization for mobile deployment
- REST API for integration
- Batch processing capabilities
- Enhanced data augmentation techniques

## License

MIT License - Free for educational and commercial use

## Team

- Siddart Singh (202310101360182)
- Adesh Srivastava (202310101360181)
- Vikash Maurya (202310101360194)

## References

- Dataset: Kaggle Fruit and Vegetable Image Recognition
- Architecture: MobileNetV2 (Sandler et al., 2018)
- Framework: TensorFlow/Keras
- Interface: Gradio

## Version History

**Current Version**: 1.0
- Complete training and inference pipeline
- Pre-trained model included
- Comprehensive documentation
- Local and cloud deployment support

---

For detailed instructions, refer to the respective documentation files listed above.
