import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def load_data(path):
    data = pd.read_csv(path)
    
    print(data.info())
    print(data.isnull().sum())

    for col in data.columns:
        num_zeros = (data[col]==0).sum()
        print(col, num_zeros)

    invalid_zero=["RestingBP", "Cholesterol"]
    data[invalid_zero]=data[invalid_zero].replace(0, np.nan)

    print(data.isnull().sum())
    return data

def save_data(data):
    pd.DataFrame(data).to_csv('../data/preprocessed_data.csv', index=False)


if __name__ == '__main__':
    data = load_data('../data/heart.csv')
    save_data(data)

