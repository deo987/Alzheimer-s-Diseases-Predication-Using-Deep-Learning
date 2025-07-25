import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import os
import argparse

# Define constants
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 4

# Function to create CNN model
def create_cnn_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)):
    """
    Create and compile a CNN model for Alzheimer's detection.

    Parameters:
        input_shape (tuple): The shape of the input images.

    Returns:
        model (tensorflow.keras.Model): The compiled CNN model.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')  # Softmax for multi-class classification
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to load data
def load_data(train_dir, val_dir):
    """
    Load and preprocess data using ImageDataGenerator.

    Parameters:
        train_dir (str): Path to the training data directory.
        val_dir (str): Path to the validation data directory.

    Returns:
        train_generator (tensorflow.keras.preprocessing.image.DirectoryIterator): Training data generator.
        val_generator (tensorflow.keras.preprocessing.image.DirectoryIterator): Validation data generator.
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

# Function to extract features using CNN model
def extract_features(model, data_generator):
    """
    Extract features from the CNN model.

    Parameters:
        model (tensorflow.keras.Model): The trained CNN model.
        data_generator (tensorflow.keras.preprocessing.image.DirectoryIterator): Data generator.

    Returns:
        features (np.array): Extracted features from the CNN model.
    """
    features = model.predict(data_generator)
    return features.reshape(features.shape[0], -1)

# Function to train machine learning models
def train_ml_models(X_train, y_train):
    """
    Train machine learning models using extracted features.

    Parameters:
        X_train (np.array): Training features.
        y_train (np.array): Training labels.

    Returns:
        models (dict): Dictionary of trained models.
    """
    models = {
        'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', objective='multi:softmax', num_class=NUM_CLASSES),
        'Gaussian Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='linear', probability=True)
    }

    for name, model in models.items():
        print(f"Training {name} model...")
        model.fit(X_train, y_train)
    
    return models

# Function to evaluate models
def evaluate_models(models, X_val, y_val, val_generator):
    """
    Evaluate the performance of trained models.

    Parameters:
        models (dict): Dictionary of trained models.
        X_val (np.array): Validation features.
        y_val (np.array): Validation labels.
        val_generator (tensorflow.keras.preprocessing.image.DirectoryIterator): Validation data generator.

    Returns:
        results (dict): Dictionary of evaluation results for each model.
    """
    results = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)  # For ROC and PR curve calculation
        
        # Classification Report
        class_report = classification_report(y_val, y_pred, target_names=list(val_generator.class_indices.keys()), output_dict=True)
        
        # Confusion Matrix
        cm = confusion_matrix(y_val, y_pred)

        # Store Results
        results[model_name] = {
            'classification_report': class_report,
            'confusion_matrix': cm
        }

        # Plot Confusion Matrix
        plot_confusion_matrix(cm, model_name, val_generator)
        
        # ROC Curve
        plot_roc_curve(y_val, y_prob, model_name)
        
        # Precision-Recall Curve
        plot_precision_recall_curve(y_val, y_prob, model_name)
    
    return results

# Function to plot confusion matrix
def plot_confusion_matrix(cm, model_name, val_generator):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Function to plot ROC curve
def plot_roc_curve(y_val, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_val, y_prob[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

# Function to plot Precision-Recall curve
def plot_precision_recall_curve(y_val, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_val, y_prob[:, 1])

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'Precision-Recall curve for {model_name}')
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.show()

# Main function
def main(args):
    # Load data
    train_generator, val_generator = load_data(args.train_dir, args.val_dir)

    # Create and train CNN model
    cnn_model = create_cnn_model()
    cnn_model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

    # Extract features from the CNN model
    cnn_features_train = extract_features(cnn_model, train_generator)
    cnn_features_val = extract_features(cnn_model, val_generator)

    # Train ML models
    models = train_ml_models(cnn_features_train, train_generator.classes)

    # Evaluate ML models
    results = evaluate_models(models, cnn_features_val, val_generator.classes, val_generator)

    # Save classification results to JSON
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=4)

    print("Results saved to", args.results_file)

if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Hybrid Approach for Alzheimer's Detection using CNN and ML Models")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the training data directory.')
    parser.add_argument('--val_dir', type=str, required=True, help='Path to the validation data directory.')
    parser.add_argument('--results_file', type=str, default='classification_results.json', help='File to save classification results.')
    
    args = parser.parse_args()
    main(args)