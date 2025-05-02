# File: /network-anomaly-detection/network-anomaly-detection/src/model_evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred):
    """
    Evaluate the performance of a machine learning model using various metrics.

    Parameters:
    y_true (array-like): True labels of the dataset.
    y_pred (array-like): Predicted labels from the model.

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    return metrics

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot the confusion matrix for the model predictions.

    Parameters:
    y_true (array-like): True labels of the dataset.
    y_pred (array-like): Predicted labels from the model.
    classes (list): List of class names for the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.show()