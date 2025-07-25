import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import os

# Constants
IMG_HEIGHT, IMG_WIDTH = 128, 128  # Image size for input
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4

# Directory paths for the dataset
TRAIN_DIR = r"C:\Users\User\OneDrive\Desktop\Alzymer\train"
VAL_DIR = r"C:\Users\User\OneDrive\Desktop\Alzymer\test"

# Data Generators
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

# Define Transfer Learning Model using ResNet50
def create_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """
    Create a transfer learning model using ResNet50 as the base.

    Args:
        input_shape (tuple): The shape of the input image.
    
    Returns:
        model: Compiled Keras transfer learning model.
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model
    
    # Add custom layers on top of the base model
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(NUM_CLASSES, activation='softmax')(x)  # Output layer with softmax activation for multi-class classification
    
    model = Model(inputs=base_model.input, outputs=x)
    return model

# Optimizer & Learning Rate Configurations
def get_optimizers():
    """
    Returns a dictionary of optimizers with different learning rates.
    """
    return {
        "Adam_0.01": tf.keras.optimizers.Adam(learning_rate=0.01),
        "Adam_0.001": tf.keras.optimizers.Adam(learning_rate=0.001),
        "Adam_0.0001": tf.keras.optimizers.Adam(learning_rate=0.0001),
        "SGD_0.01": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        "SGD_0.001": tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
        "SGD_0.0001": tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9),
        "RMSprop_0.01": tf.keras.optimizers.RMSprop(learning_rate=0.01),
        "RMSprop_0.001": tf.keras.optimizers.RMSprop(learning_rate=0.001),
        "RMSprop_0.0001": tf.keras.optimizers.RMSprop(learning_rate=0.0001),
    }

# Function to save and log classification results
def save_classification_results(results):
    """
    Save classification results to a JSON file.

    Args:
        results (dict): Dictionary containing model results.
    """
    with open("classification_results.json", "w") as f:
        json.dump(results, f, indent=4)

# Function to save the best model based on validation accuracy
def save_best_models(best_models):
    """
    Save the best models based on their validation accuracy.

    Args:
        best_models (list): List containing the best models.
    """
    for idx, best_model in enumerate(best_models):
        model_filename = f"best_model_{idx+1}.h5"
        best_model['model'].save(model_filename)
        print(f"Best Model {idx+1} saved as: {model_filename}")

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
        y_prob (array): Predicted probabilities for each class.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.show()

# Main training and evaluation function
def train_and_evaluate():
    """
    Train and evaluate the model using various optimizers and plot results.
    """
    # Load Data Generators
    train_generator, val_generator = prepare_data_generators(TRAIN_DIR, VAL_DIR)

    # Get optimizers
    optimizers = get_optimizers()

    results = {}
    best_models = []

    # Train and Evaluate Each Model
    for name, optimizer in optimizers.items():
        print(f"Training with {name}...")
        
        # Initialize and compile model
        model = create_model()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(val_generator)
        
        # Get predictions
        val_labels = val_generator.classes
        val_predictions = np.argmax(model.predict(val_generator), axis=1)

        # Classification Report
        class_report = classification_report(val_labels, val_predictions, target_names=list(val_generator.class_indices.keys()), output_dict=True)
        
        # Confusion Matrix
        cm = confusion_matrix(val_labels, val_predictions)

        # Store results
        results[name] = {
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'classification_report': class_report
        }

        # Save best models
        if len(best_models) < 2:
            best_models.append({'name': name, 'model': model, 'val_accuracy': val_accuracy})
        else:
            min_acc = min(best_models, key=lambda x: x['val_accuracy'])
            if val_accuracy > min_acc['val_accuracy']:
                best_models.remove(min_acc)
                best_models.append({'name': name, 'model': model, 'val_accuracy': val_accuracy})

    # Save Results and Best Models
    save_classification_results(results)
    save_best_models(best_models)

    # Confusion Matrix for Best Model
    best_model = best_models[0]['model']
    val_labels = val_generator.classes
    val_predictions = np.argmax(best_model.predict(val_generator), axis=1)

    cm = confusion_matrix(val_labels, val_predictions)
    plot_confusion_matrix(cm, val_generator.class_indices.keys())

    # ROC Curve for Best Model
    val_probabilities = best_model.predict(val_generator)
    plot_roc_curve(val_labels, val_probabilities)

    # Precision-Recall Curve for Best Model
    plot_precision_recall_curve(val_labels, val_probabilities)

    # Print Best Model Performance
    for best_model in best_models:
        print(f"Best Model: {best_model['name']}")
        print(f"Validation Accuracy: {best_model['val_accuracy']:.4f}")

# Execute the training and evaluation process
if _name_ == "_main_":
    train_and_evaluate()