import os
import time
# ignoring TF info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import random
import numpy as np
import tensorflow as tf

import argparse

from data_loader import load_dataset
from train import train
from test import test


DATASETS = ['cardio', 'mammo', 'satellite', 'seismic', 'annthyroid', 'thyroid', 'vowels', 'yeast']

def get_execute_time(start_time, end_time):
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f} ---".format(int(hours), int(minutes), seconds))

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def run_experiment(args):
    """
    Running a specific experiment with the given args
    Parameters
    ----------
    args: argparse args. The args given to the program
    """

    # setting seed
    seed_everything(1234)

    # loading the preprocessed dataset
    print(f"--- Loading preprocessed {args.dataset_name.capitalize()} dataset ---")
    start_time = time.time()
    dataset_X, dataset_y, siamese_pairs, features_dim, folded_train_datasets_list, folded_test_datasets_list = load_dataset(args)
    end_time = time.time()
    print(f"--- {args.dataset_name.capitalize()} dataset ready after: ", end='')
    get_execute_time(start_time, end_time)

    # training
    print("--- Start training ---")
    start_time = time.time()
    trained_estimators_list, trained_siamese_network, euclidean_nn_model, siamese_nn_model = train(dataset_X, dataset_y, siamese_pairs, folded_train_datasets_list, features_dim, args)
    end_time = time.time()
    print("--- Training finished after: ", end='')
    get_execute_time(start_time, end_time)

    # testing
    print("--- Start testing ---")
    start_time = time.time()
    test(dataset_X, folded_test_datasets_list, trained_estimators_list, trained_siamese_network, euclidean_nn_model, siamese_nn_model, args)
    end_time = time.time()
    print("--- Testing finished after: ", end='')
    get_execute_time(start_time, end_time)
    print(f"--- Finished dataset {args.dataset_name.capitalize()} ---")

if __name__ == '__main__':

    # getting the dataset to preprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', "--dataset", required=True, dest='dataset_name', type=str, help='the dataset to preprocess and save to disk for later use')
    parser.add_argument("-n", "--neighbors", dest="num_neighbors", default=10, type=int, help="The number of neighbors to retrieve from the Neareset Neighbors model")
    parser.add_argument("-a", "--augmentations", dest="num_augmentations", default=7, type=int, help="The number test-time augmentations to apply on every test sample")
    parser.add_argument("-c", "--cuml", dest="with_cuml", default=True, type=bool, help='Whether of not to use cuML')
    parser.add_argument("-s", "--siamesebatchsize", dest="siamese_batch_size", default=64, type=int, help="Batch size used to train the Siamese network")
    parser.add_argument("-b", "--siameseecpohs", dest="siamese_n_epochs", default=10, type=int, help="Number of epochs to train with the Siamese netowrk")
    parser.add_argument("-e", "--epochs", dest="num_epochs", default=500, type=int, help="The numberof epochs to train with the anomaly detector model")
    parser.add_argument("-f", "--folds", dest="n_folds", default=10, type=int, help="The number of folds in the K-Fold cross-validation")
    args = parser.parse_args()

    # adding support for runnning all of the available datasets
    if args.dataset_name == "all":
        dataset_index = int(os.environ['SLURM_ARRAY_TASK_ID'])-1
        args.dataset_name = DATASETS[dataset_index]
    
    # running experiments with the given program arguments
    run_experiment(args)
