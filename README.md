# 🛡️ Network Traffic Anomaly Detection using Hybrid and Ensemble Machine Learning Models

This project presents a robust framework for detecting anomalies in network traffic using a **Hybrid Adaptive Anomaly Detection (HAAD)** algorithm and an ensemble of machine learning models (LightGBM, XGBoost, SVM, Random Forest). It addresses class imbalance using GANs, SMOTE, and ADASYN, ensuring improved detection accuracy, especially for rare anomalies.
## 📁 Project Structure
.
├── .venv/ # Virtual environment
├── network-anomaly-det/ # Main project folder
│ ├── src/ # Source files
│ │ ├── check_imbalance.py
│ │ ├── combine_datasets.py
│ │ ├── data_preprocessing.py
│ │ ├── generate_imbalance.py
│ │ ├── generate_synthetic.py
│ │ ├── gradient_boosting_model.py
│ │ ├── haad.py
│ │ ├── model_evaluation.py
│ │ ├── run_evaluation.py
│ │ ├── utils.py
│ │ └── verify_synthetic.py
│ ├── combined_dataset.csv
│ ├── imbalanced_dataset.csv
│ ├── encoder.keras
│ └── decoder.keras
└── README.md

## 🚀 Features
- 🔍 **Anomaly Detection** using HAAD and Ensemble Learning.
- ⚖️ **Class Imbalance Handling** using GAN, SMOTE, and ADASYN.
- 📊 **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, AUC.
- 🧠 **Models Used**: Autoencoder, LightGBM, SVM, XGBoost, Random Forest.
- 🧪 **Tuned via** Grid Search and Bayesian Optimization.
- 📈 **Visual performance analysis** with ROC and PR curves.

## 🧰 Installation

To get started, clone the repository and install dependencies:
git clone https://github.com/Himanshi-1503/network-anomaly-detection.git
cd network-anomaly-detection
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

⚙️ How to Run
Execute the following scripts in sequence to run the project:
python src/combine_datasets.py
python src/data_preprocessing.py
python src/generate_synthetic.py
python src/haad.py
python src/gradient_boosting_model.py
python src/run_evaluation.py

📚 Dataset
Source: The dataset used in this project is synthetically generated.

Preprocessing Includes:
Missing value handling
Normalization
Feature selection via PCA & RFE

🧪 Results
Metric	Score
Accuracy	92.37%
Precision	98.37%
Recall	87.44%
F1-Score	92.58%
AUC	0.9364

🔮 Future Enhancements
Deploy real-time detection with Flask/FastAPI.
Streaming integration via Kafka/Spark.
Add explainability with SHAP/LIME.

👩‍💻 Author
Himanshi Sahu
