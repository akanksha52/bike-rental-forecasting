import pandas as pd

def drop_columns(df, features_to_drop):
    """ Drop irrelevant features from the dataset """ 
    return df.drop(features_to_drop, axis=1)

