import pandas as pd
import numpy as np
import pathlib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import yaml
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path):
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully from {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from {path}: {e}")
        sys.exit(1)

def data_split(data, test_size, random_state):
    try:
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info("Data split successfully into train and test sets")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        sys.exit(1)

def scale_data(X_train, X_test):
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Data scaled successfully")
        return X_train_scaled, X_test_scaled
    except Exception as e:
        logger.error(f"Failed to scale data: {e}")
        sys.exit(1)

def train_model(X_train, y_train):
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        sys.exit(1)

def save_model(model, path):
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved successfully to {path}")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        sys.exit(1)

def main():
    try:
        curr_dir = pathlib.Path(__file__)
        home_dir = curr_dir.parent.parent.parent
        params_file = home_dir.as_posix() + '/params.yaml'
        params = yaml.safe_load(open(params_file))["train_model"]
        
        input_file = '/data/external/train_data.csv'
        data_path = home_dir.as_posix() + input_file
        model_path = home_dir.as_posix() + '/models/trained_model.pkl'
        
        data = load_data(data_path)
        X_train, X_test, y_train, y_test = data_split(data, params['test_split'], params['seed'])
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        model = train_model(X_train_scaled, y_train)
        save_model(model, model_path)
    except Exception as e:
        logger.error(f"Failed to execute main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
