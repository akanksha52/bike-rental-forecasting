import pandas as pd
import numpy as np

def feature_modelling(df):
    """ Adding features for better estimation. """
    df['datetime']=pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    df['hour']=df.index.hour
    df['dayofweek']=df.index.dayofweek
    df['month']=df.index.month
    df['year']=df.index.year
    df['hour_sin']=np.sin((2*np.pi*df['hour'])/24)
    df['hour_cos']=np.cos((2*np.pi*df['hour'])/24)
    df['month_sin']=np.sin((2*np.pi*df['month'])/12)
    df['month_cos']=np.cos((2*np.pi*df['month'])/12)
    df['is_weekend']=df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_peak_commute']=df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return df
    
def encode_categoricals(df, categorical_cols):
    """ One-hot encode categorical features."""
    categorical_cols=[col for col in categorical_cols if col in df.columns]
    if not categorical_cols:
        return df 
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)
