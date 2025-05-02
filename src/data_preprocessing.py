# data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the dataset from a specified file path."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the dataset by handling missing values and duplicates."""
    data = data.drop_duplicates()
    data = data.fillna(method='ffill')  # Forward fill for missing values
    return data

def normalize_data(data):
    """Normalize the features in the dataset."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns)

def split_data(data, target_column, test_size=0.2, random_state=42):
    """Split the dataset into training and testing sets."""
    X = data.drop(columns=[target_column])
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def preprocess_data(file_path, target_column):
    """Load, clean, normalize, and split the dataset."""
    data = load_data(file_path)
    data = clean_data(data)
    data = normalize_data(data)
    X_train, X_test, y_train, y_test = split_data(data, target_column)
    return X_train, X_test, y_train, y_test