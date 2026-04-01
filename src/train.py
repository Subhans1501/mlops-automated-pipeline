import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os

# --- FIX 1: Set up MLflow tracking to use SQLite database ---
# This prevents the "meta.yaml" folder corruption and makes the UI work instantly.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Phishing_Email_Detection")

def load_data(file_path):
    """Loads the dataset and handles basic text preprocessing."""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # --- FIX 2: Smart Column Selection ---
        # We will try the most common names for email text and labels.
        # If your CSV uses different names, change these two variables:
        text_column_name = 'text_combined'   # Try changing to 'text', 'Email Text', or 'v2' if it fails
        label_column_name = 'label' # Try changing to 'label', 'Class', or 'v1' if it fails
        
        # Fallback: if those exact names aren't found, just grab the first two columns!
        if text_column_name not in df.columns:
            text_column_name = df.columns[0] 
        if label_column_name not in df.columns:
            label_column_name = df.columns[1] 
            
        print(f"Using '{text_column_name}' for emails and '{label_column_name}' for labels.")
        
        X = df[text_column_name].fillna('') 
        y = df[label_column_name]
        
        # Safety check: If labels are words like 'spam'/'ham', convert them to 1 and 0
        if y.dtype == 'object':
            y = y.astype('category').cat.codes
            
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model_params):
    """Trains a model, evaluates it, and logs everything to MLflow."""
    
    with mlflow.start_run(run_name=model_name):
        print(f"\nTraining {model_name}...")
        
        # 1. Initialize the model based on the name
        if model_name == "Logistic Regression":
            model = LogisticRegression(**model_params)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(**model_params)
        else:
            raise ValueError("Unsupported model type")

        # 2. Train the model
        model.fit(X_train, y_train)

        # 3. Make predictions
        y_pred = model.predict(X_test)

        # 4. Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"{model_name} Metrics: Accuracy={acc:.4f}, F1={f1:.4f}")

        # 5. Log everything to MLflow
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log the model itself
        mlflow.sklearn.log_model(model, artifact_path="model")
        
        print(f"Successfully logged {model_name} to MLflow.")

if __name__ == "__main__":
    # --- FIX 3: Use a raw string (r"...") to prevent Windows path escape character errors ---
    DATA_PATH = r"D:\Study\Semester 6\MLOps\Assignment_2\mlops-automated-pipeline\data\phishing_email.csv"
    
    # Load data
    X, y = load_data(DATA_PATH)

    # Convert text to numerical features using TF-IDF
    print("Vectorizing text data with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # --- Run 1: Logistic Regression ---
    lr_params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
    train_and_evaluate(X_train, X_test, y_train, y_test, "Logistic Regression", lr_params)

    # --- Run 2: Random Forest ---
    rf_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    train_and_evaluate(X_train, X_test, y_train, y_test, "Random Forest", rf_params)
    
    # --- Run 3: Logistic Regression (Different Hyperparameters) ---
    lr_params_2 = {"C": 0.1, "max_iter": 1000, "random_state": 42}
    train_and_evaluate(X_train, X_test, y_train, y_test, "Logistic Regression", lr_params_2)

    print("\nAll runs complete! Run 'mlflow ui' in your terminal to see the results.")