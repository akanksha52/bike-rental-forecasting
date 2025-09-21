import pandas as pd
import os

def load_data(path):
    """ Load CSV file into DataFrame. """
    return pd.read_csv(path)

def save_data(df, path):
    """ Save DataFrame to CSV. """
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    df.to_csv(path, index=False)  