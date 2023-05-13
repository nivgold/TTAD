from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedKFold


def load_dataset(args):
    """
    Loading dataset to be ready for training and testing

    Parameters
    ----------
    args: argparse args. The args to the program
    """

    dataset_X, dataset_y, siamese_pairs, features_dim = load_preprocessed(args.dataset_name)
    folded_train_datasets_list, folded_test_datasets_list = make_folded_datasets(dataset_X, dataset_y, args.n_folds)

    return dataset_X, dataset_y, siamese_pairs, features_dim, folded_train_datasets_list, folded_test_datasets_list

def load_preprocessed(dataset_name):
    """
    Loading saved preprocessed dataset files

    Parameters
    ----------
    dataset_name: str. The name of the preprocessed dataset to load
    """
    dataset_name = dataset_name.capitalize()    
    disk_path = '../data/' + dataset_name + "/"

    # loading dataset features and labels
    features = np.load(disk_path + f'{dataset_name}_features.npy')
    labels = np.load(disk_path + f'{dataset_name}_labels.npy')
    # loading Siamese training pairs
    siamese_pairs_X = np.load(disk_path + f'{dataset_name}_pairs_X.npy')
    siamese_pairs_y = np.load(disk_path + f'{dataset_name}_pairs_y.npy')

    # getting the dimensionality of the dataset
    features_dim = features.shape[-1]

    return features, labels, (siamese_pairs_X, siamese_pairs_y), features_dim


def minmax_normalizer(df, features):
    """
    Performing Min-Max scaling

    Parameters
    ----------
    df: pandas DataFrame. The given dataset
    features: list. The names of the dataset's features columns
    """

    # min-max normalization
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    return df


def make_folded_datasets(X, y, n_folds):
    """
    Splitting the datasets to k folds

    Parameters
    ----------
    X. pandas DataFrame. The dataset's features
    y. pandas DataFrame. The dataset's labels
    n_folds. int. The number of folds to split the dataset
    """

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    def _gen():
        """
        Generator function that is used to construct every k-fold's split dataset
        """

        for step, (train_index, test_index) in enumerate(skf.split(X, y)):
            train_features, train_labels = X[train_index], y[train_index]

            # filtering anomalous samples from the training data
            train_normal_mask = train_labels == 0

            train_features, train_labels = train_features[train_normal_mask], train_labels[train_normal_mask]
            test_features, test_labels = X[test_index], y[test_index]

            scaler = MinMaxScaler()
            train_features = scaler.fit_transform(train_features)
            test_features = scaler.transform(test_features)

            yield train_features, train_labels, test_features, test_labels
        
    # creating the folded dataset that contains all the train and test folds
    folded_dataset = tf.data.Dataset.from_generator(_gen, (tf.float64, tf.int16, tf.float64, tf.int16))

    # adding seperatly the train folds and test folds to lists
    folded_train_datasets_list = []
    folded_test_datasets_list = []
    for X_train, y_train, X_test, y_test in folded_dataset:
        train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .cache()
            .batch(32)
        )
        
        test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
            .cache()
            .batch(32)
        )
        
        # define the train ds and test ds lists
        folded_train_datasets_list.append(train_ds)
        folded_test_datasets_list.append(test_ds)
    
    return folded_train_datasets_list, folded_test_datasets_list