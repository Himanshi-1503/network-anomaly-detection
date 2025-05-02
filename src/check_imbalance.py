import pandas as pd

# Load the dataset
df = pd.read_csv('c:\\Users\\himan\\OneDrive\\Desktop\\network-anomaly-detection-1\\imbalanced_dataset.csv', header=None)

# Assuming the class label is in the last column
class_counts = df.iloc[:, -1].value_counts()
total_count = len(df)

# Calculate the percentage of each class
class_percentages = (class_counts / total_count) * 100

print("Class Distribution:")
print(class_percentages)