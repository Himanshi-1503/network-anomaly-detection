import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import networkx as nx

# Step 1: Load and preprocess the dataset
df = pd.read_csv('c:\\Users\\himan\\OneDrive\\Desktop\\network-anomaly-detection-1\\combined_dataset.csv', header=0)
df = df.drop(columns=['Unnamed: 11'], errors='ignore')
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Split the dataset into features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Step 3: Graph-based feature extraction using NetworkX
def graph_feature_extraction(X):
    G = nx.Graph()
    for i in range(len(X)):
        G.add_node(i, features=X[i])
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            weight = np.linalg.norm(X[i] - X[j])
            G.add_edge(i, j, weight=weight)
    return nx.to_numpy_array(G)

graph_features = graph_feature_extraction(X_resampled)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(graph_features, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 5: Autoencoder for anomaly detection
def build_autoencoder(input_dim):
    inputs = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(inputs)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    outputs = Dense(input_dim, activation='sigmoid')(decoded)
    return Model(inputs, outputs)

autoencoder = build_autoencoder(X_train.shape[1])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Reconstruction error for anomaly detection
reconstruction = autoencoder.predict(X_test)
reconstruction_error = np.mean(np.square(X_test - reconstruction), axis=1)

# Step 6: Reinforcement Learning for adaptive threshold adjustment
class RLAgent:
    def __init__(self, initial_threshold=0.5):
        self.threshold = initial_threshold
        self.learning_rate = 0.1

    def update_threshold(self, reward):
        self.threshold += self.learning_rate * reward
        self.threshold = max(0, min(1, self.threshold))  # Keep threshold between 0 and 1

rl_agent = RLAgent()
threshold = rl_agent.threshold

# Classify anomalies based on the threshold
y_pred = (reconstruction_error > threshold).astype(int)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

print("HAAD Algorithm - Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Step 8: ROC Curve
fpr, tpr, _ = roc_curve(y_test, reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('HAAD Algorithm - Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()