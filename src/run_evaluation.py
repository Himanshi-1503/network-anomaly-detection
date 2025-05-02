import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_evaluation import evaluate_model, plot_confusion_matrix

# Load the imbalanced dataset from CSV
df = pd.read_csv('imbalanced_dataset.csv')

# Load synthetic anomalies
synthetic_anomalies = np.load('synthetic_anomalies.npy')

# Create a DataFrame for synthetic anomalies
synthetic_df = pd.DataFrame(synthetic_anomalies, columns=df.columns[:-1])
synthetic_df['Class'] = 1  # Label synthetic anomalies as class 1

# Combine real and synthetic data
df_combined = pd.concat([df, synthetic_df], ignore_index=True)

# Count the number of occurrences of each class
class_counts = df_combined['Class'].value_counts()

# Calculate percentage imbalance
total_samples = len(df_combined)
class_percentage = (class_counts / total_samples) * 100

print("Class distribution in percentage:")
print(class_percentage)

# Split the dataset into features and target
X = df_combined.drop(columns=['Class'])
y = df_combined['Class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
# metrics = evaluate_model(y_test, y_pred)
# print("Evaluation Metrics:", metrics)

# Plot the confusion matrix
# classes = ['Class 0', 'Class 1']
# plot_confusion_matrix(y_test, y_pred, classes)