import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import argparse
import os

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4

# Define CNN Model
def create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """
    Create a CNN model for multi-class classification.

    Args:
        input_shape (tuple): The shape of the input image.
    
    Returns:
        model: Compiled Keras CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')  # Output layer with softmax activation for multi-class classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Prepare Image Generators for Data Loading
def prepare_data_generators(train_dir, val_dir):
    """
    Prepare ImageDataGenerators for training and validation datasets.

    Args:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.

    Returns:
        tuple: Training and validation data generators.
    """
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical'
    )
    return train_generator, val_generator

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm (array): Confusion matrix.
        class_names (list): List of class names for axis labels.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_true, y_prob):
    """
    Plot ROC curve for a multi-class classification model.

    Args:
        y_true (array): True labels.
        y_prob (array): Predicted probabilities for each class.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)  # Assuming binary classification or using class 1
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_true, y_prob):
    """
    Plot Precision-Recall curve.

    Args:
        y_true (array): True labels.
        y_prob (array): Predicted probabilities