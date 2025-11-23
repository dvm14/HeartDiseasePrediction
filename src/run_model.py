import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import yaml
from joblib import dump, load
import wandb
import os

def split_data(data, test_size, random_state):
    X = data.drop("HeartDisease", axis = 1)
    y = data["HeartDisease"]

    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def process_data(X_train, X_test):
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)

    return X_tr_scaled, X_te_scaled


def load_saved_data(url):
    return pd.read_csv(url)


def train_lr_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    path = os.path.abspath('../models/lr_model.pkl')
    dump(model, path)
    wandb.save(path)
    return model


def test_model(model, X_test, y_test):
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def train_rf_model(X_train, y_train, n_estimators, max_depth, random_state):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    path = os.path.abspath('../models/rf_model.pkl')
    dump(model, path)
    wandb.save(path)
    return model

def tune_rf_model(X_train, y_train):
    params = {'min_samples_leaf':[1,3,10],'n_estimators':[100,1000],
          'max_features':[0.1,0.5,1.],'max_samples':[0.5,None]}
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_, grid_search.best_score_)
    path = os.path.abspath('../models/best_rf_model.pkl')
    dump(model, path)
    wandb.save(path)
    return grid_search.best_estimator_

if __name__ == '__main__':
    data = load_saved_data('https://storage.googleapis.com/heartdiseaseprediction_bucket/data_folder/preprocessed_data.csv')
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    wandb.init(entity="dmirji", project="heart-disease-prediction", config=config)

    X_train, X_test, y_train, y_test = split_data(data, config['data']['test_size'], config['data']['random_state'])
    X_train, X_test = process_data(X_train, X_test)

    lr_model = train_lr_model(X_train, y_train)
    lr_acc = test_model(lr_model, X_test, y_test)
    print(lr_acc)
    wandb.log({"lr_accuracy": lr_acc})

    rf_model = train_rf_model(X_train, y_train, config['model']['n_estimators'], config['model']['max_depth'], config['model']['random_state'])
    rf_acc = test_model(rf_model, X_test, y_test)
    print(rf_acc)
    wandb.log({"rf_accuracy": rf_acc})

    best_rf_model = tune_rf_model(X_train, y_train)
    best_rf_acc = test_model(best_rf_model, X_test, y_test)
    print(best_rf_acc)
    wandb.log({"best_rf_accuracy": best_rf_acc})

    wandb.finish()
    