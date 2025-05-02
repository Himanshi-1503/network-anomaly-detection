import pandas as pd
import numpy as np

# Load the real dataset
df = pd.read_csv('c:\\Users\\himan\\OneDrive\\Desktop\\network-anomaly-detection-1\\imbalanced_dataset.csv', header=None)

# Load synthetic anomalies
synthetic_anomalies = np.load('synthetic_anomalies.npy')

# Create a DataFrame for synthetic anomalies
synthetic_df = pd.DataFrame(synthetic_anomalies, columns=df.columns[:-1])
synthetic_df['Class'] = 1  # Label synthetic anomalies as class 1

# Combine real and synthetic data
df_combined = pd.concat([df, synthetic_df], ignore_index=True)

# Save the combined dataset
df_combined.to_csv('c:\\Users\\himan\\OneDrive\\Desktop\\network-anomaly-detection-1\\combined_dataset.csv', header=False, index=False)

print("Combined dataset saved to combined_dataset.csv")