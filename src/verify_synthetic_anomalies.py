import numpy as np

# Load synthetic anomalies
synthetic_anomalies = np.load('synthetic_anomalies.npy')

# Display the shape and first few rows of the synthetic anomalies
print("Shape of synthetic anomalies:", synthetic_anomalies.shape)
print("First few rows of synthetic anomalies:")
print(synthetic_anomalies[:5])