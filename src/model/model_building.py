import pandas as pd
import numpy as np
import yaml
import os
import xgboost as xgb
import pickle

def get_params(params_file: str) -> dict:
    try:
        with open(params_file, "r") as f:
            params = yaml.safe_load(f)
        return params["model-building"]
    except FileNotFoundError as e:
        print(f"Params file not found: {e}")
        raise
    except KeyError as e:
        print(f"Missing key in params file: {e}")
        raise
    except Exception as e:
        print(f"Error loading params: {e}")
        raise

def fetch_data(data_path: str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(os.path.join(data_path, "train_bow.csv"))
        return train_data
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        raise
    except Exception as e:
        print(f"Error reading data: {e}")
        raise

def model_training(train_data: pd.DataFrame, params: dict) -> xgb.XGBClassifier:
    try:
        X_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        xgb_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_estimators=params.get("n_estimators", 100),
            eta=params.get("learning_rate", 0.3)
        )
        xgb_model.fit(X_train, y_train)
        return xgb_model
    except Exception as e:
        print(f"Error during model training: {e}")
        raise

def save_model(xgb_model: xgb.XGBClassifier, model_path: str) -> None:
    try:
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, "xgb_model.pkl")
        with open(model_file, "wb") as f:
            pickle.dump(xgb_model, f)
        print("Model saved to", model_file)
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

def main():
    try:
        params = get_params("params.yaml")
        train_data = fetch_data(os.path.join("data", "features"))
        model_path = os.path.join("src", "models")
        xgb_model = model_training(train_data, params)
        save_model(xgb_model, model_path)
        print("Model training pipeline completed successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()