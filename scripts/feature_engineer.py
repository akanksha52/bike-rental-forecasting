import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class feature_engineer:
    
    def __init(self):
        self.scaler=StandardScaler()
        self.feature_names=None
        
    def load_and_split_data(self):
        print("Loading and splitting data...")
        train_df=pd.read_csv('../data/raw/train.csv')
        test_df=pd.read_csv('..data/raw/test.csv')
        X=train_df.drop('count', axis=1)
        y=train_df['count']
        X_train, y_train, X_val, y_val=train_test_split(X, y, test_size=0.2, random_state=40)
        print("Data split complete:")
        print(f"Train: {X_train.shape[0]} samples")
        print(f"Validation: {X_val.shape[0]} samples") 
        print(f"Test: {test_df.shape[0]} samples")
        return X_train, X_val, test_df, y_train, y_val 
    
    def safe_feature_selection(self, df):
        columns_to_remove=['casual', 'registered', 'atemp', 'workingday']
        df_clean=df.drop(columns_to_remove, axis=1, errors='ignore')
        available_features=['season', 'holiday', 'weather', 'temp', 'humidity', 'windspeed']
        X=df_clean[available_features]
        y=df_clean['count']
        return X, y, available_features
    
    def engineer_features(self, fit_scaler=True):
        print("🛠 Starting feature engineering...")
        X_train, X_val, X_test, y_train, y_val=self.load_and_split_data()
        X_train_clean, y_train, feature_names=self.safe_feature_selection(pd.concat([X_train, y_train], axis=1))
        X_val_clean, y_val, _=self.safe_feature_selection(pd.concat([X_val, y_val], axis=1))
        X_test_clean, y_test, _=self.safe_feature_selection(X_test)
        self.feature_names=feature_names
        print(f"✅ Selected features: {feature_names}")
        if fit_scaler:
            X_train_scaled=self.scaler.fit_transform(X_train_clean)
            X_val_scaled=self.scaler.transform(X_val_clean)
            X_test_scaled=self.scaler.transform(X_test_clean)
        else:
            X_train_scaled=self.scaler.transform(X_train_clean)
            X_val_scaled=self.scaler.transform(X_val_clean) 
            X_test_scaled=self.scaler.transform(X_test_clean)
        self.save_processed_data(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)
        print("✅ Feature engineer
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        pd.DataFrame(X_train).to_csv('../data/processed/X_train.csv', index=False)
        pd.DataFrame(X_val).to_csv('../data/processed/X_val.csv', index=False) 
        pd.DataFrame(X_test).to_csv('../data/processed/X_test.csv', index=False)
        y_train.to_csv('../data/processed/y_train.csv', index=False)
        y_val.to_csv('../data/processed/y_val.csv', index=False)
        
        if y_test is not None and len(y_test)>0:
            y_test.to_csv('y_test.csv', index=False)
        
        # Save scaler and feature names
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, MODELS_DIR / 'feature_scaler.pkl')
        
        print("💾 Processed data saved to data/processed/")

def main():
    """Run feature engineering pipeline"""
    engineer = FeatureEngineer()
    engineer.engineer_features()

if __name__ == "__main__":
    main()
        
    
    

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
# from scripts.config import *
