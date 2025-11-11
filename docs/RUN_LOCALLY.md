# Local Installation Guide

This document provides instructions for running the Fruit & Vegetable Classifier on your local machine.

## System Requirements

- Python 3.8 or higher
- 4 GB RAM minimum (8 GB recommended)
- 2 GB available disk space
- Internet connection for initial setup

## Installation Steps

### 1. Verify Python Installation

Ensure Python is installed and accessible:
```bash
python --version
```

If Python is not installed, download it from https://www.python.org/downloads/ and ensure "Add Python to PATH" is selected during installation.

### 2. Install Dependencies

Navigate to the project directory and install required packages:
```bash
pip install -r requirements.txt
```

For systems with limited resources, install without cache:
```bash
pip install -r requirements.txt --no-cache-dir
```

### 3. Verify Model File

Confirm that `FV.h5` (approximately 17 MB) exists in the project directory. This pre-trained model is required for inference.

## Running the Application

### Start the Web Interface

Execute the following command:
```bash
python run_app.py
```

The application will start a local server at http://127.0.0.1:7860 and should open automatically in your default browser.

### Using the Interface

1. Navigate to http://127.0.0.1:7860 if the browser doesn't open automatically
2. Click the upload area to select an image
3. The model will process the image and display predictions with confidence scores
4. Results include the top prediction and category classification (fruit or vegetable)

## Project Files

- `run_app.py` - Main application script
- `train_model.py` - Training script for custom models
- `FV.h5` - Pre-trained model weights
- `requirements.txt` - Python package dependencies
- `dataset/` - Directory for training data (if training custom models)

## Troubleshooting

### Python Not Found
Verify Python installation and PATH configuration. Reinstall Python if necessary, ensuring PATH is configured correctly.

### Dependency Installation Failures
Update pip before installing dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Port Conflicts
If port 7860 is already in use, modify `run_app.py`:
```python
server_port=7861  # Change to an available port
```

### Model Loading Errors
Ensure `FV.h5` exists in the same directory as `run_app.py`. The file should be approximately 17 MB.

## Supported Classifications

The model recognizes 36 classes:

**Fruits (15)**: Apple, Banana, Bell Pepper, Chilli Pepper, Grapes, Jalepeno, Kiwi, Lemon, Mango, Orange, Paprika, Pear, Pineapple, Pomegranate, Watermelon

**Vegetables (21)**: Beetroot, Cabbage, Capsicum, Carrot, Cauliflower, Corn, Cucumber, Eggplant, Garlic, Ginger, Lettuce, Onion, Peas, Potato, Radish, Soy Beans, Spinach, Sweet Corn, Sweet Potato, Tomato, Turnip

## Performance Notes

- First run may take longer due to dependency loading
- Subsequent runs start faster as packages are cached
- Application runs offline after initial setup
- Stop the server with Ctrl+C in the terminal

## Additional Resources

For training custom models, refer to `TRAINING_GUIDE.md`.  
For complete command reference, see `COMMANDS.md`.  
For project overview, consult `README.md`.
