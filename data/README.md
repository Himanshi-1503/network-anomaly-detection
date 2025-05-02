# README for Data Directory

# Data Directory Structure

This directory contains the datasets used for the network anomaly detection project. It is organized into two main subdirectories:

## 1. `raw/`
This directory holds the raw dataset files that are used for network anomaly detection. These datasets have not been modified or processed and are in their original format.

## 2. `processed/`
This directory contains the processed datasets that are ready for analysis and model training. The data in this directory has undergone various preprocessing steps, including cleaning, normalization, and feature extraction.

# Data Sources

The datasets used in this project are sourced from publicly available repositories, including the KDD Cup 1999 dataset, which is widely used for evaluating anomaly detection algorithms in network traffic.

# Preprocessing Steps

The preprocessing of raw data involves several key steps:
- **Data Cleaning:** Removing any irrelevant or corrupted entries from the dataset.
- **Normalization:** Scaling the features to ensure that they contribute equally to the analysis.
- **Feature Extraction:** Selecting and transforming relevant features that enhance the model's ability to detect anomalies.

For detailed preprocessing code, please refer to the Jupyter notebook located in the `notebooks/` directory.