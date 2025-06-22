import numpy as np
import pandas as pd

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import os

def safe_nltk_download(resource):
    try:
        nltk.download(resource)
    except Exception as e:
        print(f"Error downloading NLTK resource '{resource}': {e}")

safe_nltk_download('wordnet')
safe_nltk_download('stopwords')

def fetch_data(data_path: str) -> tuple:
    try:
        train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(data_path, "test.csv"))
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        raise

def lemmatization(text):
    try:
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(y) for y in text]
        return " ".join(text)
    except Exception as e:
        print(f"Error in lemmatization: {e}")
        return text

def remove_stop_words(text):
    try:
        stop_words = set(stopwords.words("english"))
        Text = [i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    except Exception as e:
        print(f"Error removing stop words: {e}")
        return text

def removing_numbers(text):
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        print(f"Error removing numbers: {e}")
        return text

def lower_case(text):
    try:
        text = text.split()
        text = [y.lower() for y in text]
        return " ".join(text)
    except Exception as e:
        print(f"Error converting to lower case: {e}")
        return text

def removing_punctuations(text):
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        print(f"Error removing punctuations: {e}")
        return text

def removing_urls(text):
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        print(f"Error removing URLs: {e}")
        return text

def remove_small_sentences(df):
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        print(f"Error removing small sentences: {e}")

def normalize_text(df):
    try:
        df.content = df.content.apply(lambda content: lower_case(content))
        df.content = df.content.apply(lambda content: remove_stop_words(content))
        df.content = df.content.apply(lambda content: removing_numbers(content))
        df.content = df.content.apply(lambda content: removing_punctuations(content))
        df.content = df.content.apply(lambda content: removing_urls(content))
        df.content = df.content.apply(lambda content: lemmatization(content))
        return df
    except Exception as e:
        print(f"Error normalizing text: {e}")
        raise

def get_processed_data(data_path: str) -> tuple:
    try:
        train_data, test_data = fetch_data(data_path)
        train_processed = normalize_text(train_data)
        test_processed = normalize_text(test_data)
        return train_processed, test_processed
    except Exception as e:
        print(f"Error processing data: {e}")
        raise

def save_data(data_path: str, train_processed: pd.DataFrame, test_processed: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_processed.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
    except Exception as e:
        print(f"Error saving processed data: {e}")
        raise

def main():
    try:
        data_path = os.path.join("data", "raw")
        train_data, test_data = get_processed_data(data_path)
        data_path = os.path.join("data", "processed")
        save_data(data_path, train_data, test_data)
        print("Data processing completed successfully.")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()