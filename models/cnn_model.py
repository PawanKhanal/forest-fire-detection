"""CNN model architecture for forest fire detection."""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from typing import Tuple, Optional, Dict, Any
import os
import json

class ForestFireCNN:
    """Convolutional Neural Network for fire detection from images."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        learning_rate: float = 0.001
    ) -> None:
        """
        Initialize CNN model.
        
        Args:
            input_shape: Input image dimensions (height, width, channels)
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        
    def build_model(self) -> tf.keras.Model:
        """
        Build and compile the CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model
    
    def train(
        self,
        train_generator,
        validation_generator,
        epochs: int = 20,
        class_weight: Optional[Dict[int, float]] = None,
        callbacks: Optional[list] = None
    ) -> tf.keras.callbacks.History:
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            class_weight: Optional class weights for imbalanced data
            callbacks: Optional list of Keras callbacks
            
        Returns:
            Training history object
            
        Raises:
            ValueError: If model is not built before training
        """
        if self.model is None:
            self.build_model()
        
        if callbacks is None:
            callbacks = self._get_default_callbacks()
        
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            class_weight=class_weight,
            callbacks=callbacks
        )
        
        return self.history
    
    def _get_default_callbacks(self) -> list:
        """Create default training callbacks."""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/checkpoints/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
    
    def predict(self, image_batch: tf.Tensor) -> tf.Tensor:
        """
        Make predictions on image batch.
        
        Args:
            image_batch: Batch of preprocessed images
            
        Returns:
            Prediction probabilities
        """
        return self.model.predict(image_batch)
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model
        """
        self.model.save(filepath)
        
    @classmethod
    def load(cls, filepath: str) -> 'ForestFireCNN':
        """
        Load saved model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded ForestFireCNN instance
        """
        instance = cls()
        instance.model = models.load_model(filepath)
        return instance