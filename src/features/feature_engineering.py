import pandas as pd
import numpy as np
import yaml
import os
from sklearn.feature_extraction.text import CountVectorizer

def get_params(params_file: str) -> dict:
    try:
        with open(params_file, "r") as f:
            params = yaml.safe_load(f)
        return params["feature-engineering"]
    except FileNotFoundError as e:
        print(f"Params file not found: {e}")
        raise
    except KeyError as e:
        print(f"Missing key in params file: {e}")
        raise
    except Exception as e:
        print(f"Error loading params: {e}")
        raise

def fetch_data(data_path: str) -> tuple:
    try:
        train_data = pd.read_csv(os.path.join(data_path, "train_processed.csv"))
        test_data = pd.read_csv(os.path.join(data_path, "test_processed.csv"))
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        raise
    except Exception as e:
        print(f"Error reading data: {e}")
        raise

def splitting_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple:
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values
        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        return X_train, y_train, X_test, y_test
    except KeyError as e:
        print(f"Missing column in data: {e}")
        raise
    except Exception as e:
        print(f"Error splitting data: {e}")
        raise

def bag_of_words(X_train, X_test, y_train, y_test, max_features: int) -> tuple:
    try:
        vectorizer = CountVectorizer(max_features=max_features)
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)
        train_df = pd.DataFrame(X_train_bow if isinstance(X_train_bow, np.ndarray) else X_train_bow.toarray())
        train_df['label'] = y_train
        test_df = pd.DataFrame(X_test_bow if isinstance(X_test_bow, np.ndarray) else X_test_bow.toarray())
        test_df['label'] = y_test
        return train_df, test_df
    except Exception as e:
        print(f"Error in Bag of Words transformation: {e}")
        raise

def save_data(data_path: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"), index=False)
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"), index=False)
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

def main():
    try:
        params = get_params("params.yaml")
        train_data, test_data = fetch_data(os.path.join("data", "processed"))
        train_data.fillna('', inplace=True)
        test_data.fillna('', inplace=True)
        X_train, y_train, X_test, y_test = splitting_data(train_data, test_data)
        train_df, test_df = bag_of_words(X_train, X_test, y_train, y_test, params["max_features"])
        data_path = os.path.join("data", "features")
        save_data(data_path, train_df, test_df)
        print("Feature engineering completed successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()