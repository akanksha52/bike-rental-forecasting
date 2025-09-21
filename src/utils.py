import pandas as pd

def load_data(path):
    """ Load CSV file into DataFrame. """
    return pd.read_csv(path)

def save_data(df, path):
    """ Save DataFrame to CSV. """
    df.to_csv(path, index=False)
