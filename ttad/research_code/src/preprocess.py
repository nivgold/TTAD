from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import argparse

DATASET_DATA_FILE = {
    'Cardio':
        {
            'data': 'cardio.csv',
            'label_col': '21'
        },
    'Mammo':
        {
            'data': 'mammo.csv',
            'label_col': '6'
        },
    'Satellite':
        {
            'data': 'satellite.csv',
            'label_col': '36'
        },
    'Seismic':
        {
            'data': 'seismic.csv',
            'label_col': '11'
        },
    'Annthyroid':
        {
            'data': 'annthyroid.csv',
            'label_col': '6'
        },
    'Thyroid':
        {
            'data': 'thyroid.csv',
            'label_col': '6'
        },
    'Vowels':
        {
            'data': 'vowels.csv',
            'label_col': '12'
        },
    'Yeast':
        {
            'data': 'yeast.csv',
            'label_col': '8'
        },
    'Satimage':
        {
            'data': 'satimage.csv',
            'label_col': '36'
        },
    'Smtp':
        {
            'data': 'smtp.csv',
            'label_col': '3'
        },
    'Wine':
        {
            'data': 'wine.csv',
            'label_col': '13'
        }
}

def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and modify the data type to reduce memory usage.

    Parameters:
    df: pandas Dataframe. A preprocessed dataset
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('--- Memory usage of dataframe is {:.2f} MB ---'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('--- Memory usage after optimization is: {:.2f} MB ---'.format(end_mem))
    print('--- Decreased by {:.1f}% ---'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def preprocessing(dataset_name):
    """
    Preprocessing the given dataset

    Parameters
    ----------
    dataset_name: str. The name of the given dataset
    """

    # retrieving essential info on the given dataset
    data_file = DATASET_DATA_FILE[dataset_name]['data']
    label_col = DATASET_DATA_FILE[dataset_name]['label_col']

    data_file_path = '../data/' + dataset_name + '/' + data_file
    
    print(f"--- Openning {dataset_name} ---")
    df = pd.read_csv(data_file_path)

    # getting dataset's features
    features = list(set(df.columns) - set([label_col]))

    # replacing inf values with column max
    df = replace_inf_values(df, features)

    # reducing memory consumption
    df = reduce_mem_usage(df)

    dataset_labels = df[label_col]
    dataset_features = df.drop(columns=[label_col], axis=1)
    siamese_pairs_X, siamese_pairs_y = generate_pairs(dataset_features)

    # saving preprocessed datasaet and Siamese pairs to disk
    save_to_disk(dataset_features, dataset_labels, siamese_pairs_X, siamese_pairs_y, dataset_name)

def replace_inf_values(df, features):
    """
    Replacing inf values in the features columns with max value of every feature

    Parameters
    ----------
    df: pandas DataFrame. The given dataset
    features: list. The names of the dataset's features columns
    """

    df[features] = df[features].replace(np.inf, np.nan)
    df[features] = df[features].fillna(df[features].max())

    return df

def generate_pairs(data):
    """
    Generating pairs for Siamese-netowrk training.
    The pairs constructed contain 50% same-class pairs and 50% different-class pairs.

    Parameters
    ----------
    data: pandas DataFrame. The dataset's features
    """

    # using Isolation Forest as label propagating
    print("--- Using Isolation Forest for Siamese pairs creation ---")
    normal_data, anomaly_data = isolation_forest_labeling(data)

    left_list = []
    right_list = []
    label_list = []

    print("--- Pairing from normal data ---")
    for idx, row in tqdm(normal_data.iterrows(), total=len(normal_data)):
        same_sample = normal_data.sample(n=1, random_state=42).iloc[0]
        left_list.append(row)
        right_list.append(same_sample)
        label_list.append(1)
        different_sample = anomaly_data.sample(n=1, random_state=1234).iloc[0]
        left_list.append(row)
        right_list.append(different_sample)
        label_list.append(0)
    
    print("--- Pairing from anomaly data ---")
    for idx, row in tqdm(anomaly_data.iterrows(), total=len(anomaly_data)):
        same_sample = anomaly_data.sample(n=1, random_state=43).iloc[0]
        left_list.append(row)
        right_list.append(same_sample)
        label_list.append(1)
        different_sample = normal_data.sample(n=1, random_state=1235).iloc[0]
        left_list.append(row)
        right_list.append(different_sample)
        label_list.append(0)

    left_list_data = pd.DataFrame(left_list).reset_index(drop=True)
    left_list_data.columns = [f'left_{col}' for col in left_list_data.columns]

    right_list_data = pd.DataFrame(right_list).reset_index(drop=True)
    right_list_data.columns = [f'right_{col}' for col in right_list_data.columns]

    label_df = pd.DataFrame({'label': label_list})

    full_df = pd.concat([left_list_data, right_list_data, label_df], axis=1)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

    left_pairs = full_df.loc[:, left_list_data.columns]
    right_pairs = full_df.loc[:, right_list_data.columns]

    X = np.array([left_pairs.values, right_pairs.values])
    y = full_df['label'].values

    return X, y

def isolation_forest_labeling(data):
    """
    Using Isolation Forest to do label propagation.

    Parameters
    ----------
    data: pandas DataFrame. The dataset's features
    """

    data = data.copy()

    # creating Isolation Forest Classifier
    clf = IsolationForest(n_estimators=200, max_samples='auto', contamination=float(.12),
                          max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
    
    clf.fit(data)
    data['anomaly_score'] = clf.score_samples(data) * -1
    
    percent_high = np.percentile(data['anomaly_score'], 95)
    percent_low = np.percentile(data['anomaly_score'], 40)
    
    # seperating normal and anomalous instances using trained Isolation Forest
    normal_data = data[data['anomaly_score'] < percent_low].drop(columns=['anomaly_score'], axis=1)
    anomaly_data = data[data['anomaly_score'] >= percent_high].drop(columns=['anomaly_score'], axis=1)

    return normal_data, anomaly_data

def save_to_disk(features, labels, siamese_pairs_X, siamese_pairs_y, dataset_name):
    """
    Saving preprocessed data and pairs to disk

    Parameters
    ----------
    features: pandas DataFrame. The dataset features
    labels: pandas DataFrame. The dataset labels
    siamese_pairs_X: numpy ndarray. The siamese pairs' features
    siamese_pairs_y: numpy ndarray. The siamese pairs' labels. Either 0 or 1 (same or different class)
    """

    disk_path = "../data/" + dataset_name + "/"

    # save preprocessed dataset
    np.save(disk_path + f'{dataset_name}_features.npy', features)
    np.save(disk_path + f'{dataset_name}_labels.npy', labels)
    # save Siamese pairs
    np.save(disk_path + f'{dataset_name}_pairs_X.npy', siamese_pairs_X)
    np.save(disk_path + f'{dataset_name}_pairs_y.npy', siamese_pairs_y)

if __name__ == '__main__':

    # getting the dataset to preprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, dest='dataset_name', type=str, help='the dataset to preprocess and save to disk for later use')
    args = parser.parse_args()

    dataset_name = args.dataset_name.capitalize()

    if dataset_name not in DATASET_DATA_FILE.keys():
        raise ValueError("Provided dataset is not supported")

    print(f"--- Starting preprocess {dataset_name} dataset ---")
    preprocessing(dataset_name)
