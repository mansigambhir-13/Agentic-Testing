"""
GTZAN Music Genre Classification - CNN Model
=============================================
Advanced deep learning model for genre classification using spectrogram images
This script builds a Convolutional Neural Network (CNN) to classify music genres
using the mel-spectrogram images from the GTZAN dataset.

Author: Your Name
Date: November 2024
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check and install required deep learning dependencies"""
    required = {
        'tensorflow': 'tensorflow',
        'keras': 'included with tensorflow',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'cv2': 'opencv-python'
    }
    
    print("Checking deep learning dependencies...")
    for module, install_name in required.items():
        try:
            if module == 'keras':
                import tensorflow.keras
            elif module == 'PIL':
                from PIL import Image
            elif module == 'cv2':
                import cv2
            elif module == 'sklearn':
                import sklearn
            else:
                __import__(module)
            print(f"✓ {module} is installed")
        except ImportError:
            if install_name != 'included with tensorflow':
                print(f"Installing {module}...")
                os.system(f"{sys.executable} -m pip install {install_name}")
            else:
                print(f"  {module} is {install_name}")

def prepare_image_data(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and prepare spectrogram images for CNN training
    
    Args:
        dataset_path: Path to the dataset directory
    
    Returns:
        Tuple of (images, labels, class_names)
    """
    from PIL import Image
    import cv2
    
    print("\n" + "=" * 60)
    print("Preparing Image Data")
    print("=" * 60)
    
    # Find images directory
    images_path = dataset_path / "Data" / "images_original"
    if not images_path.exists():
        images_path = dataset_path / "images_original"
    
    if not images_path.exists():
        print("Could not find images_original directory")
        print("Please ensure the dataset is properly downloaded")
        return None, None, None
    
    # Get all genre directories
    genres = [d.name for d in images_path.iterdir() if d.is_dir()]
    genres.sort()
    print(f"\nFound {len(genres)} genres: {', '.join(genres)}")
    
    # Load images
    images = []
    labels = []
    image_size = (128, 128)  # Resize images to standard size
    
    print("\nLoading spectrogram images...")
    for genre_idx, genre in enumerate(genres):
        genre_path = images_path / genre
        genre_images = list(genre_path.glob("*.png")) + list(genre_path.glob("*.jpg"))
        
        print(f"  Loading {genre}: {len(genre_images)} images", end="")
        
        for img_path in genre_images[:100]:  # Limit to 100 per genre for memory
            try:
                # Load and preprocess image
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.resize(img, image_size)
                    img = img / 255.0  # Normalize to [0, 1]
                    images.append(img)
                    labels.append(genre_idx)
            except Exception as e:
                print(f"    Error loading {img_path.name}: {e}")
        
        print(f" ✓")
    
    # Convert to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Image shape: {images[0].shape if len(images) > 0 else 'N/A'}")
    print(f"Classes: {genres}")
    
    return images, labels, genres

def build_cnn_model(input_shape: Tuple[int, int, int], num_classes: int):
    """
    Build a CNN model for music genre classification
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of genre classes
    
    Returns:
        Compiled Keras model
    """
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
            BatchNormalization, GlobalAveragePooling2D
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
        
        print("\n" + "=" * 60)
        print("Building CNN Model")
        print("=" * 60)
        
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, 
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            GlobalAveragePooling2D(),
            Dropout(0.5),
            
            # Fully Connected Layers
            Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.5),
            
            # Output Layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        return model
        
    except ImportError:
        print("TensorFlow/Keras not installed. Install with: pip install tensorflow")
        return None

def train_model(model, X_train, y_train, X_val, y_val, epochs: int = 50):
    """
    Train the CNN model
    
    Args:
        model: Keras model
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        epochs: Number of training epochs
    
    Returns:
        Training history
    """
    try:
        from tensorflow.keras.callbacks import (
            EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        )
        
        print("\n" + "=" * 60)
        print("Training CNN Model")
        print("=" * 60)
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✓ Model training complete!")
        
        return history
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None

def evaluate_model(model, X_test, y_test, class_names: List[str]):
    """
    Evaluate the trained model and create visualizations
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        class_names: List of genre names
    """
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        import seaborn as sns
        
        print("\n" + "=" * 60)
        print("Evaluating Model")
        print("=" * 60)
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.3f}")
        print(f"Test Loss: {test_loss:.3f}")
        
        # Get predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - CNN Model')
        plt.xlabel('Predicted Genre')
        plt.ylabel('Actual Genre')
        plt.tight_layout()
        
        viz_dir = Path("gtzan_cnn_results")
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / 'cnn_confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: cnn_confusion_matrix.png")
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, per_class_acc)
        plt.xlabel('Genre')
        plt.ylabel('Accuracy')
        plt.title('Per-Genre Classification Accuracy')
        plt.xticks(rotation=45)
        plt.ylim([0, 1])
        for i, v in enumerate(per_class_acc):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
        plt.tight_layout()
        plt.savefig(viz_dir / 'per_genre_accuracy.png', dpi=100, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved: per_genre_accuracy.png")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

def plot_training_history(history):
    """
    Plot training history curves
    
    Args:
        history: Keras training history object
    """
    if history is None:
        return
    
    print("\n" + "=" * 60)
    print("Training History")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    viz_dir = Path("gtzan_cnn_results")
    viz_dir.mkdir(exist_ok=True)
    plt.savefig(viz_dir / 'training_history.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(f"✓ Saved: training_history.png")

def main():
    """Main execution function"""
    print("╔" + "=" * 58 + "╗")
    print("║" + " GTZAN CNN Music Genre Classifier ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Check dependencies
    check_dependencies()
    
    try:
        import kagglehub
        from sklearn.model_selection import train_test_split
        
        # Download dataset
        print("\nDownloading GTZAN dataset...")
        dataset_path = kagglehub.dataset_download(
            "andradaolteanu/gtzan-dataset-music-genre-classification"
        )
        dataset_path = Path(dataset_path)
        print(f"✓ Dataset path: {dataset_path}")
        
        # Load and prepare image data
        images, labels, class_names = prepare_image_data(dataset_path)
        
        if images is None:
            print("\nError: Could not load image data")
            return
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"\nData split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
        
        # Build model
        model = build_cnn_model(
            input_shape=X_train[0].shape,
            num_classes=len(class_names)
        )
        
        if model is None:
            print("\nError: Could not build model")
            return
        
        # Train model
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=30)
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test, class_names)
        
        print("\n" + "=" * 60)
        print("CNN Training Complete!")
        print("=" * 60)
        print("\nModel and results saved to: ./gtzan_cnn_results/")
        print("\nNext steps:")
        print("1. Try data augmentation (rotation, zoom, shift)")
        print("2. Experiment with transfer learning (VGG16, ResNet)")
        print("3. Implement ensemble methods")
        print("4. Try different spectrogram parameters")
        print("5. Add attention mechanisms to the model")
        
    except Exception as e:
        print(f"\nError in main execution: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install kagglehub tensorflow pillow opencv-python scikit-learn")

if __name__ == "__main__":
    main()

