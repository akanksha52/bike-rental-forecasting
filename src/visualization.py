import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df):
    """ Plot correlation among various features """
    numeric_df=df.select_dtypes(include=['float64', 'int64'])
    corr_matrix=numeric_df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cbar=True, square=True)
    plt.title("Correlation Heatmap of Numerical Features with count", fontsize=14, pad=12)
    plt.show()

def plot_distribution(df, col): 
    """ Plot histogram for col features """
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()