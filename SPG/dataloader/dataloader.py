import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from utility.utils import get_dataloaders

import sys

def get_data(params):
    if params['combined']:
        d1_name, d1_path = params['d1_name'], params['d1_path']
        d2_name, d2_path = params['d2_name'], params['d2_path']
        d3_name, d3_path = params['d3_name'], params['d3_path']
        params1 = {'dataset_name': d1_name, 'dataset_path': d1_path}
        params2 = {'dataset_name': d2_name, 'dataset_path': d2_path}
        params3 = {'dataset_name': d3_name, 'dataset_path': d3_path}

        d1, _ = get_dataset(params1)
        d2, _ = get_dataset(params2)
        d3, _ = get_dataset(params3)

        return get_dataloaders(params, d1, d2, d3)
    else:
        df, params = get_dataset(params)
        return get_dataloaders(params, df)

def get_dataset(params):
    dataset_name = params['dataset_name']

    if dataset_name == 'pump_sensor':
        df, params = get_pump_sensor_data(params)
    elif dataset_name == 'occupancy':
        df, params = get_occupancy_data(params)
    elif dataset_name == 'machine_failure':
        df, params = get_machine_failure_data(params)
    elif dataset_name == 'elevator_failure':
        df, params = get_elevator_failure_data(params)
    elif dataset_name == 'smart_home':
        df, params = get_smart_home_data(params)
    elif dataset_name == 'agricultural':
        df, params = get_agricultural_data(params)
    elif dataset_name == 'head_posture':
        df, params = get_head_posture_data(params)
    else:
        print('Unknown dataset')
        sys.exit(0)

    return df, params


def get_pump_sensor_data(params):
    path_df = params['dataset_path']
    df = pd.read_csv(path_df)
    df.drop(columns=['Unnamed: 0', 'timestamp', 'machine_status', 'sensor_15'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    params['n_feats'] = df.shape[1]

    return df, params


def get_occupancy_data(params):
    path_df = params['dataset_path']
    df = pd.read_csv(path_df)
    df.drop(columns=['date', 'Occupancy'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    params['n_feats'] = df.shape[1]

    return df, params


def get_machine_failure_data(params):
    path_df = params['dataset_path']
    df = pd.read_csv(path_df)
    df.drop(columns=['UDI', 'Product ID', 'Type', 'Target', 'Failure Type'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    params['n_feats'] = df.shape[1]

    return df, params

def get_elevator_failure_data(params):
    path_df = params['dataset_path']
    df = pd.read_csv(path_df)
    df.drop(columns=['ID'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    params['n_feats'] = df.shape[1]

    return df, params

def get_smart_home_data(params):
    path_df = params['dataset_path']
    df = pd.read_csv(path_df, low_memory=False)
    df.drop(columns=['time', 'icon', 'summary', 'cloudCover'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    params['n_feats'] = df.shape[1]

    return df, params


def get_agricultural_data(params):
    path_df = params['dataset_path']
    df = pd.read_csv(path_df)
    df.drop(columns=['Timestamp'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    params['n_feats'] = df.shape[1]

    return df, params

def get_head_posture_data(params):
    path_df = params['dataset_path']
    df = pd.read_csv(path_df)
    df.drop(columns=['Miscare', 'Time'], inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    params['n_feats'] = df.shape[1]

    return df, params









