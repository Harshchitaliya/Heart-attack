import pandas as pd
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
import yaml
import sys

def load_data(path):
    df = pd.read_csv(path)
    return df

def data_split(data, test_size, random_state):
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Combine X and y for train and test data to return
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    return train_data, test_data

def save(train_data, test_data, outpath):
    pathlib.Path(outpath).mkdir(parents=True, exist_ok=True)
    train_data.to_csv(outpath + '/train_data.csv', index=False)
    test_data.to_csv(outpath + '/test_data.csv', index=False)

def main():
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir.as_posix() + '/params.yaml'
    params = yaml.safe_load(open(params_file))["make_dataset"]

    input_file = '/data/raw/heart.csv'
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/data/external'
    
    data = load_data(data_path)
    train_data, test_data = data_split(data, params['test_split'], params['seed'])
    save(train_data, test_data, output_path)

if __name__ == "__main__":
    main()
