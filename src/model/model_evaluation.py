import numpy as np
import pandas as pd
import pickle
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def fetch_data(data_path: str) -> pd.DataFrame:
    try:
        test_data = pd.read_csv(os.path.join(data_path, "test_bow.csv"))
        return test_data
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        raise
    except Exception as e:
        print(f"Error reading data: {e}")
        raise

def make_predictions(model_path: str, X_test_bow: np.ndarray) -> tuple:
    try:
        with open(os.path.join(model_path, "xgb_model.pkl"), "rb") as f:
            xgb_model = pickle.load(f)
        y_pred = xgb_model.predict(X_test_bow)
        y_pred_proba = xgb_model.predict_proba(X_test_bow)[:, 1]
        return y_pred, y_pred_proba
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        raise
    except Exception as e:
        print(f"Error making predictions: {e}")
        raise

def cal_evaluation_metrics(y_test: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc
        }
        return metrics_dict
    except Exception as e:
        print(f"Error calculating evaluation metrics: {e}")
        raise

def save_metrics(metrics_dict: dict, metrics_path: str) -> None:
    try:
        os.makedirs(metrics_path, exist_ok=True)
        metrics_file = os.path.join(metrics_path, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Metrics saved to {metrics_file}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
        raise

def main():
    try:
        test_data = fetch_data("data/features")
        X_test_bow = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values
        y_pred, y_pred_proba = make_predictions("models", X_test_bow)
        metrics_dict = cal_evaluation_metrics(y_test, y_pred, y_pred_proba)
        save_metrics(metrics_dict, "reports/metrics")
        print("Evaluation pipeline completed successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()