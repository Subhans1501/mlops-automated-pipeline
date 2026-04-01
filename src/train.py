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
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Phishing_Email_Detection")
def load_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df=pd.read_csv(file_path)
        text_column_name='text_combined'
        label_column_name='label'
        if text_column_name not in df.columns:
            text_column_name=df.columns[0] 
        if label_column_name not in df.columns:
            label_column_name=df.columns[1] 
        print(f"Using '{text_column_name}' for emails and '{label_column_name}' for labels.")
        X=df[text_column_name].fillna('') 
        y=df[label_column_name]
        if y.dtype=='object':
            y=y.astype('category').cat.codes
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model_params):
    with mlflow.start_run(run_name=model_name):
        print(f"\nTraining {model_name}...")
        if model_name == "Logistic Regression":
            model = LogisticRegression(**model_params)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(**model_params)
        else:
            raise ValueError("Unsupported model type")
        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred,zero_division=0)
        recall=recall_score(y_test,y_pred,zero_division=0)
        f1=f1_score(y_test,y_pred,zero_division=0)
        print(f"{model_name} Metrics: Accuracy={acc:.4f}, F1={f1:.4f}")
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy",acc)
        mlflow.log_metric("precision",precision)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("f1_score",f1)
        mlflow.sklearn.log_model(model,artifact_path="model")
        print(f"Successfully logged {model_name} to MLflow.")
if __name__ == "__main__":
    DATA_PATH=r"data/phishing_email.csv"
    X,y=load_data(DATA_PATH)
    print("Vectorizing text data with TF-IDF...")
    vectorizer=TfidfVectorizer(max_features=5000)
    X_vectorized=vectorizer.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size=0.2,random_state=42)
    lr_params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
    train_and_evaluate(X_train, X_test, y_train, y_test, "Logistic Regression", lr_params)
    rf_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    train_and_evaluate(X_train, X_test, y_train, y_test, "Random Forest", rf_params)
    lr_params_2 = {"C": 0.1, "max_iter": 1000, "random_state": 42}
    train_and_evaluate(X_train, X_test, y_train, y_test, "Logistic Regression", lr_params_2)
    print("\nAll runs complete! Run 'mlflow ui' in your terminal to see the results.")