import data_preprocessing
import feature_engineering
import utils

df_train=utils.load_data('../data/raw/train.csv') 
df_test=utils.load_data('../data/raw/test.csv')

df_train=data_preprocessing.drop_columns(df_train, ['registered', 'casual'])

df_train=feature_engineering.feature_modelling(df_train)
df_test=feature_engineering.feature_modelling(df_test)

df_test=feature_engineering.encode_categoricals(df_train, ['season', 'weather'])
df_test=feature_engineering.encode_categoricals(df_test, ['season', 'weather'])

utils.save_data(df_train, '../data/processed/train.csv')
utils.save_data(df_test, '../data/processed/test.csv')