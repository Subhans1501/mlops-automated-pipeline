# 🚀 MLOps Automated Pipeline: Phishing Email Detection

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-Tracking_%26_Registry-0194E2.svg)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine_Learning-F7931E.svg)

An end-to-end Machine Learning Operations (MLOps) pipeline designed to automatically train, evaluate, track, and deploy a binary classification model for detecting phishing emails. 

## 📋 Project Overview
This repository demonstrates a complete CI/CD and Continuous Training (CT) lifecycle for machine learning. It features automated experiment tracking, model registry management, and scheduled retraining using GitHub Actions and MLflow.

### Key Features
* **Automated CI/CD:** GitHub Actions automatically installs dependencies, runs `pytest` suites, and executes training scripts upon every push to the `main` branch.
* **Experiment Tracking:** MLflow tracks all model parameters, hyperparameters, and evaluation metrics (Accuracy, F1, Precision, Recall).
* **Model Registry:** The best-performing model (Logistic Regression) is automatically registered and transitioned to the `Production` stage.
* **Continuous Training (CT):** A scheduled cron job triggers weekly retraining pipelines to ensure the model adapts to new data patterns without manual intervention.
* **Automated Deployment:** Integration with the Hugging Face Hub API for automated model artifact deployment.

## 📂 Repository Structure

```text
mlops-automated-pipeline/
│
├── .github/workflows/       # GitHub Actions CI/CD pipelines
│   ├── ci.yml               # Triggered on push (Test -> Train -> Deploy)
│   └── retrain.yml          # Time-based trigger (Scheduled Retraining)
│
├── data/                    # Dataset directory
│   └── phishing_email.csv   
│
├── src/                     # Source code directory
│   ├── train.py             # Main training, evaluation, and MLflow logging script
│   ├── test_model.py        # Pytest suite for data integrity checks
│   └── deploy.py            # Hugging Face deployment script
│
├── .gitignore               # Git ignore rules (excludes mlflow.db, mlruns)
├── requirements.txt         # Python package dependencies
└── README.md                # Project documentation
🛠️ Local Setup & Execution
To run this pipeline locally on your machine:

1. Clone the repository

Bash
git clone [https://github.com/](https://github.com/)[YourUsername]/mlops-automated-pipeline.git
cd mlops-automated-pipeline
2. Create and activate a virtual environment

Bash
python -m venv .venv
source .venv/Scripts/activate  # On Windows
# source .venv/bin/activate    # On Mac/Linux
3. Install dependencies

Bash
pip install -r requirements.txt
4. Run local tests

Bash
pytest src/test_model.py
5. Execute the training pipeline

Bash
python src/train.py
6. View MLflow Dashboard

Bash
mlflow ui
Navigate to http://127.0.0.1:5000 in your browser to view experiment tracking and model registries.

📈 Model Performance
Two models were evaluated during the initial pipeline run:

Logistic Regression (Production): Accuracy = 0.981, F1-Score = 0.982

Random Forest: Accuracy = 0.929, F1-Score = 0.934

Due to superior performance metrics, the Logistic Regression model was automatically selected, registered, and promoted to Production.