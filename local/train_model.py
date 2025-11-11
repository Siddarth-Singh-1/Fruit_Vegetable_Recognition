"""
Fruit & Vegetable Recognition - Complete Training Script
Train a MobileNetV2 model locally on your PC
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ============================================
# CONFIGURATION
# ============================================

# Dataset paths - MODIFY THESE TO YOUR DATASET LOCATION
DATASET_DIR = '../dataset'  # Use relative path to avoid Windows long path issues
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

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

NUM_CLASSES = len(labels)

# ============================================
# FUNCTIONS
# ============================================

def check_dataset():
    """Verify dataset structure"""
    print("=" * 60)
    print("Checking Dataset...")
    print("=" * 60)
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ ERROR: Dataset directory '{DATASET_DIR}' not found!")
        print("\nPlease ensure your dataset is organized as:")
        print("  dataset/")
        print("    ├── train/")
        print("    │   ├── apple/")
        print("    │   ├── banana/")
        print("    │   └── ...")
        print("    └── validation/")
        print("        ├── apple/")
        print("        ├── banana/")
        print("        └── ...")
        return False
    
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ ERROR: Training directory '{TRAIN_DIR}' not found!")
        return False
    
    if not os.path.exists(VAL_DIR):
        print(f"❌ ERROR: Validation directory '{VAL_DIR}' not found!")
        return False
    
    train_classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    val_classes = [d for d in os.listdir(VAL_DIR) if os.path.isdir(os.path.join(VAL_DIR, d))]
    
    print(f"✅ Dataset found!")
    print(f"   Training classes: {len(train_classes)}")
    print(f"   Validation classes: {len(val_classes)}")
    print(f"   Expected classes: {NUM_CLASSES}")
    
    if len(train_classes) != NUM_CLASSES or len(val_classes) != NUM_CLASSES:
        print(f"\n⚠️  WARNING: Number of classes doesn't match expected ({NUM_CLASSES})")
    
    return True

def create_data_generators():
    """Create data generators with augmentation"""
    print("\n" + "=" * 60)
    print("Creating Data Generators...")
    print("=" * 60)
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✅ Data generators created!")
    print(f"   Training samples: {train_generator.samples}")
    print(f"   Validation samples: {val_generator.samples}")
    print(f"   Classes found: {len(train_generator.class_indices)}")
    
    return train_generator, val_generator

def visualize_samples(train_generator):
    """Visualize sample images from training set"""
    print("\n" + "=" * 60)
    print("Visualizing Sample Images...")
    print("=" * 60)
    
    sample_images, sample_labels = next(train_generator)
    
    plt.figure(figsize=(15, 10))
    for i in range(min(9, len(sample_images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(sample_images[i])
        class_idx = np.argmax(sample_labels[i])
        plt.title(f"Class: {labels[class_idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("✅ Sample images saved as 'sample_images.png'")
    plt.close()

def build_model():
    """Build MobileNetV2 transfer learning model"""
    print("\n" + "=" * 60)
    print("Building Model...")
    print("=" * 60)
    
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n✅ Model built successfully!")
    print(f"   Total parameters: {model.count_params():,}")
    
    # Print model summary
    model.summary()
    
    return model

def setup_callbacks():
    """Setup training callbacks"""
    print("\n" + "=" * 60)
    print("Setting up Callbacks...")
    print("=" * 60)
    
    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        '../models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    callbacks = [early_stopping, reduce_lr, checkpoint]
    
    print("✅ Callbacks configured:")
    print("   - Early Stopping (patience=5)")
    print("   - Learning Rate Reduction (patience=3)")
    print("   - Model Checkpoint (best_model.h5)")
    
    return callbacks

def train_model(model, train_generator, val_generator, callbacks):
    """Train the model"""
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 60)
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save the final model
    model.save('../models/FV.h5')
    print("\n✅ Training complete!")
    print("   Model saved as '../models/FV.h5'")
    print("   Best model saved as '../models/best_model.h5'")
    
    return history

def plot_training_history(history):
    """Plot and save training history"""
    print("\n" + "=" * 60)
    print("Plotting Training History...")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("✅ Training history saved as 'training_history.png'")
    plt.close()

def evaluate_model(model, val_generator):
    """Evaluate model on validation set"""
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60)
    
    val_loss, val_accuracy = model.evaluate(val_generator, verbose=1)
    
    print(f"\n✅ Evaluation Results:")
    print(f"   Validation Loss: {val_loss:.4f}")
    print(f"   Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    
    return val_loss, val_accuracy

def test_predictions(model, val_generator):
    """Test model with sample predictions"""
    print("\n" + "=" * 60)
    print("Testing Sample Predictions...")
    print("=" * 60)
    
    val_generator.reset()
    test_images, test_labels = next(val_generator)
    
    # Make predictions
    predictions = model.predict(test_images[:9], verbose=0)
    
    # Display results
    plt.figure(figsize=(15, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_images[i])
        
        true_class = np.argmax(test_labels[i])
        pred_class = np.argmax(predictions[i])
        confidence = predictions[i][pred_class]
        
        color = 'green' if true_class == pred_class else 'red'
        plt.title(f"True: {labels[true_class]}\nPred: {labels[pred_class]}\n({confidence*100:.1f}%)", 
                  color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=300)
    print("✅ Test predictions saved as 'test_predictions.png'")
    plt.close()

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main training pipeline"""
    print("\n")
    print("=" * 60)
    print("  FRUIT & VEGETABLE RECOGNITION - TRAINING SCRIPT")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   {gpu}")
    else:
        print("⚠️  No GPU detected. Training will use CPU (slower)")
    
    # Step 1: Check dataset
    if not check_dataset():
        print("\n❌ Dataset check failed. Please fix the issues and try again.")
        return
    
    # Step 2: Create data generators
    train_generator, val_generator = create_data_generators()
    
    # Step 3: Visualize samples
    visualize_samples(train_generator)
    
    # Step 4: Build model
    model = build_model()
    
    # Step 5: Setup callbacks
    callbacks = setup_callbacks()
    
    # Step 6: Train model
    history = train_model(model, train_generator, val_generator, callbacks)
    
    # Step 7: Plot training history
    plot_training_history(history)
    
    # Step 8: Evaluate model
    evaluate_model(model, val_generator)
    
    # Step 9: Test predictions
    test_predictions(model, val_generator)
    
    # Final summary
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE!")
    print("=" * 60)
    print("Generated files:")
    print("  ✓ ../models/FV.h5 - Final trained model")
    print("  ✓ ../models/best_model.h5 - Best model during training")
    print("  ✓ training_history.png - Training curves")
    print("  ✓ sample_images.png - Sample training images")
    print("  ✓ test_predictions.png - Test predictions")
    print("\nYou can now use '../models/FV.h5' with run_app.py for predictions!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
