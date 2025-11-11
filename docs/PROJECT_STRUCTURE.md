# Project Structure

## Directory Layout

```
Fruit_Vegetable_Recognition/
│
├── Core Application Files
│   ├── run_app.py                    # Web interface for image classification
│   ├── train_model.py                # Model training script
│   └── requirements.txt              # Python package dependencies
│
├── Pre-trained Models
│   ├── FV.h5                         # Main trained model (17 MB)
│   └── best_model.h5                 # Best checkpoint from training
│
├── Documentation
│   ├── README.md                     # Project overview and features
│   ├── START_HERE.md                 # Quick start guide
│   ├── COMMANDS.md                   # Complete command reference
│   ├── RUN_LOCALLY.md                # Local installation guide
│   ├── TRAINING_GUIDE.md             # Model training documentation
│   ├── QUICK_REFERENCE.md            # Quick reference card
│   ├── PROJECT_SUMMARY.md            # Comprehensive project summary
│   └── PROJECT_STRUCTURE.md          # This file
│
├── Jupyter Notebooks
│   ├── Fruit_Vegetable_Recognition_Colab.ipynb    # Training notebook
│   └── Fruit_Vegetable_Classifier_UI.ipynb        # Inference notebook
│
├── Dataset Directory
│   └── dataset/
│       ├── train/                    # Training images by class
│       │   ├── apple/
│       │   ├── banana/
│       │   ├── beetroot/
│       │   └── ... (36 classes)
│       └── validation/               # Validation images by class
│           ├── apple/
│           ├── banana/
│           ├── beetroot/
│           └── ... (36 classes)
│
├── Training Outputs (Generated)
│   ├── training_history.png          # Training/validation curves
│   ├── sample_images.png             # Sample training data
│   └── test_predictions.png          # Model predictions visualization
│
└── Virtual Environment (Optional)
    └── venv/                         # Python virtual environment
```

## File Descriptions

### Core Scripts

**run_app.py**
- Launches Gradio web interface
- Loads pre-trained model
- Handles image uploads and predictions
- Displays results with confidence scores
- Runs on http://127.0.0.1:7860

**train_model.py**
- Complete training pipeline
- Data loading and augmentation
- Model construction with MobileNetV2
- Training with callbacks
- Performance evaluation
- Visualization generation

**requirements.txt**
- Lists all Python dependencies
- Version specifications for packages
- Used with `pip install -r requirements.txt`

### Models

**FV.h5**
- Final trained model
- Size: ~17 MB
- 36-class classifier
- Ready for inference

**best_model.h5**
- Best performing checkpoint
- Saved during training
- Used for model recovery

### Documentation Files

**README.md**
- Project overview
- Features and capabilities
- Quick start instructions
- Technical specifications

**START_HERE.md**
- Getting started guide
- Basic commands
- Initial setup steps
- Recommended workflow

**COMMANDS.md**
- Complete command reference
- Installation commands
- Execution commands
- Troubleshooting commands
- Configuration examples

**RUN_LOCALLY.md**
- Local installation guide
- System requirements
- Step-by-step setup
- Troubleshooting section

**TRAINING_GUIDE.md**
- Training documentation
- Dataset preparation
- Configuration options
- Performance optimization

**QUICK_REFERENCE.md**
- Quick command reference
- Common tasks
- Configuration tips
- Troubleshooting table

**PROJECT_SUMMARY.md**
- Comprehensive overview
- Technical specifications
- Usage instructions
- System requirements

### Jupyter Notebooks

**Fruit_Vegetable_Recognition_Colab.ipynb**
- Google Colab training notebook
- Complete training workflow
- Dataset setup options
- GPU acceleration support
- Model download functionality

**Fruit_Vegetable_Classifier_UI.ipynb**
- Google Colab inference notebook
- Model upload and loading
- Interactive web interface
- No training required

### Dataset Structure

**dataset/train/**
- Training images organized by class
- Each class in separate subdirectory
- 36 class folders total
- Images in JPG/PNG format

**dataset/validation/**
- Validation images organized by class
- Same structure as training set
- Used for model evaluation
- 36 class folders total

### Generated Files

**training_history.png**
- Training and validation accuracy curves
- Training and validation loss curves
- Generated after training completes

**sample_images.png**
- Sample images from training dataset
- Shows data augmentation effects
- Generated at training start

**test_predictions.png**
- Model predictions on test images
- Shows predicted vs actual labels
- Generated after training completes

## File Sizes

| File | Size |
|------|------|
| FV.h5 | ~17 MB |
| best_model.h5 | ~17 MB |
| requirements.txt | <1 KB |
| run_app.py | ~4 KB |
| train_model.py | ~13 KB |
| training_history.png | ~200 KB |
| sample_images.png | ~500 KB |
| test_predictions.png | ~500 KB |
| Dataset (complete) | ~1 GB |

## Key Directories

### Working Directory
All commands should be executed from the project root directory where `run_app.py` and `train_model.py` are located.

### Dataset Directory
Must contain `train/` and `validation/` subdirectories with 36 class folders each. Required for training.

### Virtual Environment
Optional `venv/` directory for isolated Python environment. Created with `python -m venv venv`.

## File Dependencies

### run_app.py requires:
- FV.h5 (model file)
- Python packages from requirements.txt
- No dataset required

### train_model.py requires:
- dataset/train/ (training images)
- dataset/validation/ (validation images)
- Python packages from requirements.txt
- Generates FV.h5 and other outputs

## Workflow Paths

### Inference Workflow
```
requirements.txt → pip install → run_app.py → FV.h5 → Web Interface
```

### Training Workflow
```
requirements.txt → pip install → dataset/ → train_model.py → FV.h5
```

### Documentation Workflow
```
START_HERE.md → README.md → Specific Guides (RUN_LOCALLY.md, TRAINING_GUIDE.md)
```

## Maintenance

### Files to Keep
- All .py scripts
- All .md documentation
- FV.h5 model
- requirements.txt
- Jupyter notebooks
- Dataset directory

### Files Safe to Delete
- best_model.h5 (if not needed)
- training_history.png (regenerated on training)
- sample_images.png (regenerated on training)
- test_predictions.png (regenerated on training)
- venv/ (can be recreated)

### Files Never to Delete
- run_app.py
- train_model.py
- FV.h5
- requirements.txt
- Documentation files

## Navigation Guide

**Want to run the classifier?**
→ See START_HERE.md or RUN_LOCALLY.md

**Want to train a model?**
→ See TRAINING_GUIDE.md

**Need specific commands?**
→ See COMMANDS.md

**Want project overview?**
→ See README.md

**Need quick reference?**
→ See QUICK_REFERENCE.md

**Want complete details?**
→ See PROJECT_SUMMARY.md
