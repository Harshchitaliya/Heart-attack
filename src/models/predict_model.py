import pandas as pd
import numpy as np
import pathlib
import joblib
from sklearn.preprocessing import StandardScaler
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

def load_model(path):
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded successfully from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model from {path}: {e}")
        sys.exit(1)

def scale_data(data, scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        data_scaled = scaler.transform(data)
        logger.info("Data scaled successfully")
        return data_scaled
    except Exception as e:
        logger.error(f"Failed to scale data: {e}")
        sys.exit(1)

def make_predictions(model, data):
    try:
        predictions = model.predict(data)
        logger.info("Predictions made successfully")
        return predictions
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        sys.exit(1)

def save_predictions(predictions, path):
    try:
        np.savetxt(path, predictions, delimiter=",")
        logger.info(f"Predictions saved successfully to {path}")
    except Exception as e:
        logger.error(f"Failed to save predictions to {path}: {e}")
        sys.exit(1)

def main():
    try:
        curr_dir = pathlib.Path(__file__)
        home_dir = curr_dir.parent.parent.parent
        
        input_file = '/data/external/test_data.csv'
        data_path = home_dir.as_posix() + input_file
        model_path = home_dir.as_posix() + '/models/trained_model.pkl'
        scaler_path = home_dir.as_posix() + '/models/scaler.pkl'
        predictions_path = home_dir.as_posix() + '/data/predictions/predictions.csv'
        
        data = load_data(data_path)
        model = load_model(model_path)
        data_scaled = scale_data(data, scaler_path)
        predictions = make_predictions(model, data_scaled)
        save_predictions(predictions, predictions_path)
    except Exception as e:
        logger.error(f"Failed to execute main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
