import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import itertools

# Load the imbalanced dataset from CSV
df = pd.read_csv('imbalanced_dataset.csv')

# Load synthetic anomalies
synthetic_anomalies = np.load('synthetic_anomalies.npy')

# Create a DataFrame for synthetic anomalies
synthetic_df = pd.DataFrame(synthetic_anomalies, columns=df.columns[:-1])
synthetic_df['Class'] = 1  # Label synthetic anomalies as class 1

# Combine real and synthetic data
df_combined = pd.concat([df, synthetic_df], ignore_index=True)

# Split the dataset into features and target
X = df_combined.drop(columns=['Class'])
y = df_combined['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate models
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    roc_auc = auc(*roc_curve(y_test, y_pred_proba)[:2]) if y_pred_proba is not None else None

    # Append results
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC": roc_auc
    })

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Class 0', 'Class 1'], title=f'{model_name} - Confusion Matrix')

    # Plot ROC Curve
    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba, model_name)

# Confusion Matrix Plotting Function
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# ROC Curve Plotting Function
def plot_roc_curve(y_test, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Store results for all models
results = []

# Train and evaluate multiple models
models = {
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

for model_name, model in models.items():
    print(f"\nTraining and Evaluating: {model_name}")
    evaluate_model(model, model_name)

# Display combined results
results_df = pd.DataFrame(results)
print("\nCombined Results:")
print(results_df)