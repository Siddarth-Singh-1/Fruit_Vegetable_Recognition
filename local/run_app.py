"""
Fruit & Vegetable Classifier - Local Application
Run this script to launch the web UI on your local PC
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import gradio as gr
import os

# Model parameters
IMG_SIZE = 224
MODEL_PATH = "../models/FV.h5"

# Class labels
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 
    5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 
    10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 
    15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 
    20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 
    25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 
    33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

def load_classifier_model():
    """Load the pre-trained model"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found!\n"
            f"Please ensure FV.h5 is in the same directory as this script."
        )
    
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Model supports {len(labels)} classes\n")
    return model

def predict_fruit_vegetable(image):
    """
    Predict fruit/vegetable from uploaded image
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Formatted prediction results
    """
    # Ensure image is PIL Image
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Resize and preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Get class name
    class_name = labels[predicted_class].capitalize()
    
    # Determine category
    fruits = ['Apple', 'Banana', 'Bell pepper', 'Chilli pepper', 'Grapes', 'Jalepeno', 
              'Kiwi', 'Lemon', 'Mango', 'Orange', 'Paprika', 'Pear', 'Pineapple', 
              'Pomegranate', 'Watermelon']
    
    category = "🍎 Fruit" if class_name in fruits else "🥬 Vegetable"
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_results = {
        labels[idx].capitalize(): float(predictions[0][idx]) 
        for idx in top_3_idx
    }
    
    # Format result
    result = f"""
### Prediction: {class_name}
**Category:** {category}  
**Confidence:** {confidence*100:.2f}%

#### Top 3 Predictions:
"""
    for name, conf in top_3_results.items():
        result += f"\n- **{name}**: {conf*100:.2f}%"
    
    return result

if __name__ == "__main__":
    print("=" * 60)
    print("🍍🍅 Fruit & Vegetable Classifier")
    print("=" * 60)
    
    # Load model
    try:
        model = load_classifier_model()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        exit(1)
    
    # Create Gradio interface
    demo = gr.Interface(
        fn=predict_fruit_vegetable,
        inputs=gr.Image(type="pil", label="Upload Fruit or Vegetable Image"),
        outputs=gr.Markdown(label="Prediction Results"),
        title="🍍🍅 Fruit & Vegetable Classifier",
        description="Upload an image of a fruit or vegetable to classify it! Supports 36 different classes.",
        theme="soft",
        allow_flagging="never",
        examples=None
    )
    
    # Launch the interface
    print("🚀 Launching interactive web UI...")
    print("📱 The interface will open in your browser automatically!")
    print("=" * 60)
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )
