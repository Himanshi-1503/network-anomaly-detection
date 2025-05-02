import pandas as pd
import numpy as np

# Creating a dataset with class imbalance
np.random.seed(42)
data_size = 1000

# 90% of class 0, 10% of class 1
classes = np.random.choice([0, 1], size=data_size, p=[0.9, 0.1])

# Creating a DataFrame with 10 features
df = pd.DataFrame({
    'Feature1': np.random.randn(data_size),
    'Feature2': np.random.randn(data_size),
    'Feature3': np.random.randn(data_size),
    'Feature4': np.random.randn(data_size),
    'Feature5': np.random.randn(data_size),
    'Feature6': np.random.randn(data_size),
    'Feature7': np.random.randn(data_size),
    'Feature8': np.random.randn(data_size),
    'Feature9': np.random.randn(data_size),
    'Feature10': np.random.randn(data_size),
    'Class': classes
})

# Save the DataFrame to a CSV file
df.to_csv('imbalanced_dataset.csv', index=False)

# Display the first few rows
print(df.head())