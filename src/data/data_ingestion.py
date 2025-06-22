import numpy as np
import pandas as pd
import yaml

import os

from sklearn.model_selection import train_test_split 


def get_params(params_file: str) -> float:
    try:
        with open(params_file, "r") as f:
            params = yaml.safe_load(f)
        return params["data-ingestion"]["test_size"]
    except FileNotFoundError:
        print(f"Error: File '{params_file}' not found.")
        raise
    except KeyError as e:
        print(f"Error: Missing key {e} in params file.")
        raise
    except Exception as e:
        print(f"Unexpected error reading params: {e}")
        raise
    
def read_data(url: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(url)
        return data
    
    except Exception as e:
        print(f"Error reading data from {url}: {e}")
        raise

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data.drop(columns=['tweet_id'],inplace=True)
        
        data.fillna('', inplace=True)

        final_df = data[data['sentiment'].isin(['neutral','sadness'])]

        final_df['sentiment'].replace({'neutral':1, 'sadness':0},inplace=True)

        return final_df
    
    except KeyError as e:
        print(f"Error: Missing column {e} in data.")
        raise
    
    except Exception as e:
        print(f"Unexpected error processing data: {e}")
        raise
    
def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

def main():
    try:
        test_size = get_params("params.yaml")
        
        data = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        
        final_df = process_data(data)
        
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        
        data_path = os.path.join("data", "raw")
        save_data(data_path, train_data, test_data)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    
    main()