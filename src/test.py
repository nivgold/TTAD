from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
import tensorflow as tf
from timeit import default_timer as timer

from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score, roc_auc_score, auc, precision_recall_curve

from tqdm import tqdm

from imblearn.over_sampling import SMOTE

from cuml.cluster import KMeans as cuKMeans
from sklearn.cluster import KMeans

from prettytable import PrettyTable

from constants import evaluated_algorithms, evaluated_estimators

# define compared algorithms from the Experiments section
# evaluated_algorithms = [
#     'WO_TTA_Baseline',
#     'Gaussian_TTA_Baseline',
#     'Euclidean_SMOTE_TTA',
#     'Siamese_SMOTE_TTA',
#     'Euclidean_Kmeans_TTA',
#     'Siamese_Kmeans_TTA'
# ]

# # define anomaly detection estimators from the Experiments section
# evaluated_estimators = [
#     'Autoencoder',
#     'Isolation Forest',
#     'One-Class SVM',
#     'Local Outlier Factor'
# ]

def test(X, folded_test_datasets_list, trained_estimators_list, trained_siamese_network, euclidean_nn_model, siamese_nn_model, args):
    """
    Performing test phase on the test set with all of the compared algorithms described in the Experiments section
    Parameters
    ----------
    X: ndarray of shape (#num_samples, #features). The dataset's features
    folded_test_datasets_list: list. The test set of each split in the k-fold
    trained_estimators_list: list. The trained estimator of each split in hte k-fold
    trained_siamese_network. TF's Model. The trained siamese internal model. used to obtain embedding of each test instance
    euclidean_nn_model: trained Neareset Neighbors model with euclidean distance metric
    siamese_nn_model: trained Neareset Neighbors model with Siamese distance metric
    args: argparse args. The args given to the program
    """

    # define k-fold split's metrics dict
    # algorithm:estimator:metric:fold
    algorithms_folds_metrics_dict = {algorithm: {estimator: [] for estimator in evaluated_estimators} for algorithm in evaluated_algorithms}

    for split_index in range(args.n_folds):
        # test set-up
        test_ds = folded_test_datasets_list[split_index]
        trained_estimator = trained_estimators_list[split_index]
        ae_test_step_func = ae_test_step()
        
        print(f"--- Testing k-fold split index: {split_index+1} ---")
        # testing current k-fold split
        current_split_algorithms_metrics = test_loop(X, test_ds, trained_estimator, euclidean_nn_model, siamese_nn_model, trained_siamese_network, ae_test_step_func, args)

        # update the folds metrics dictionary
        for algorithm, estimator_metrics in current_split_algorithms_metrics.items():
            for estimator, metrics in estimator_metrics.items():
                algorithms_folds_metrics_dict[algorithm][estimator].append(metrics)
    
    for algorithm, estimator_folds_metrics in algorithms_folds_metrics_dict.items():
        for estimator, folds_metrics in estimator_folds_metrics.items():
            algorithms_folds_metrics_dict[algorithm][estimator] = np.array(folds_metrics)
    
    # presenting results
    print_test_results(algorithms_folds_metrics_dict, args)

def test_loop(X, test_ds, trained_estimator, euclidean_nn_model, siamese_nn_model, trained_siamese_network, ae_test_step_func, args):
    """
    Performing the test loop with every evaluated algorithm.
    Parameters
    ----------
    X: numpy ndarray of shape (#num_samples, #features). The dataset's features
    test_ds: TF's Dataset. The test set
    trained_estimator: A trained anomaly detector.
    euclidean_nn_model. trained Nearest Neighbors model with euclidean distance metric
    siamese_nn_model. trained Nearest Neighbors mkodel with siamese distance metric
    trained_siamese_network: TF's Model. The trained siamese model used for calculating two samples' distance
    ae_test_step_func: function. The function that is used for performing single test step with AE anomaly detection estimator
    args: argparse args. The args given to the program
    """

    # loss function - with reduction equals to `NONE` in order to get the loss of every test example
    loss_func = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    # extract estimators
    trained_ae = trained_estimator['ae']
    # trained_if = trained_estimator['if']
    trained_ocs = trained_estimator['ocs']
    trained_lof = trained_estimator['lof']

    num_neighbors = args.num_neighbors
    num_augmentations = args.num_augmentations

    algorithms_test_loss = {algorithm: {estimator: [] for estimator in evaluated_estimators} for algorithm in evaluated_algorithms}
    algorithms_metrics = {algorithm: {estimator: None for estimator in evaluated_estimators} for algorithm in evaluated_algorithms}
    times_dict = {algorithm: {estimator: None for estimator in evaluated_estimators} for algorithm in evaluated_algorithms}

    test_labels = []
    tqdm_total_bar = test_ds.cardinality().numpy()
    for step, (x_batch_test, y_batch_test) in tqdm(enumerate(test_ds), total=tqdm_total_bar):
        # predicting with each estimator without TTA (first Baseline)
        ae_inference_ts_start = timer()
        ae_reconstruction_loss = ae_test_step_func(x_batch_test, trained_ae, loss_func).numpy()
        
        # if_inference_ts_start = timer()
        # if_anomaly_score = if_test_step(x_batch_test, trained_if)
        
        ocs_inference_ts_start = timer()
        ocs_anomaly_score = ocs_test_step(x_batch_test, trained_ocs)
        
        lof_inference_ts_start = timer()
        lof_anomaly_score = lof_test_step(x_batch_test, trained_lof)
        
        end_inference_ts = timer()

        # saving first Baseline times for each estimator
        times_dict['WO_TTA_Baseline']['Autoencoder'] = ocs_inference_ts_start - ae_inference_ts_start
        # times_dict['WO_TTA_Baseline']['Isolation Forest'] = ocs_inference_ts_start - if_inference_ts_start
        times_dict['WO_TTA_Baseline']['One-Class SVM'] = lof_inference_ts_start - ocs_inference_ts_start
        times_dict['WO_TTA_Baseline']['Local Outlier Factor'] = end_inference_ts - lof_inference_ts_start

        # saving first Baseline results for each estimator
        algorithms_test_loss['WO_TTA_Baseline']['Autoencoder'].append(ae_reconstruction_loss)
        # algorithms_test_loss['WO_TTA_Baseline']['Isolation Forest'].append(if_anomaly_score)
        algorithms_test_loss['WO_TTA_Baseline']['One-Class SVM'].append(ocs_anomaly_score)
        algorithms_test_loss['WO_TTA_Baseline']['Local Outlier Factor'].append(lof_anomaly_score)
        test_labels.append(y_batch_test.numpy())




        ######################## TTA ########################
        

        # calculate euclidean nn indices
        eaclidean_nn_ts_start = timer()
        euclidean_nn_batch_neighbors_indices = euclidean_nn_model.kneighbors(X=x_batch_test.numpy(), n_neighbors=num_neighbors, return_distance=False)
        eaclidean_nn_ts_end = timer()
        eaclidean_nn_total_time = eaclidean_nn_ts_end - eaclidean_nn_ts_start
        # calculate siamese nn indices
        test_batch_latent_features = trained_siamese_network(x_batch_test).numpy()
        siamese_nn_ts_start = timer()
        siamese_nn_batch_neighbors_indices = siamese_nn_model.kneighbors(X=test_batch_latent_features, n_neighbors=num_neighbors, return_distance=False)
        siamese_nn_ts_end = timer()
        siamese_nn_total_time = siamese_nn_ts_end - siamese_nn_ts_start
        euclidean_nn_batch_neighbors_features = X[euclidean_nn_batch_neighbors_indices]
        siamese_nn_batch_neighbors_features = X[siamese_nn_batch_neighbors_indices]


        gn_tta_samples_ts_start = timer()
        gn_tta_samples = generate_random_noise_tta_samples(x_batch_test.numpy(), num_augmentations=num_augmentations)
        
        es_tta_samples_ts_start = timer()
        es_tta_samples = generate_oversampling_tta_samples(euclidean_nn_batch_neighbors_features, oversampling_method=SMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations)
        
        ss_tta_samples_ts_start = timer()
        ss_tta_samples = generate_oversampling_tta_samples(siamese_nn_batch_neighbors_features, oversampling_method=SMOTE, num_neighbors=num_neighbors, num_augmentations=num_augmentations)
        
        ekm_tta_samples_ts_start = timer()
        ekm_tta_samples = generate_kmeans_tta_samples(euclidean_nn_batch_neighbors_features, args.with_cuml, num_augmentations=num_augmentations)
        
        skm_tta_samples_ts_start = timer()
        skm_tta_samples = generate_kmeans_tta_samples(siamese_nn_batch_neighbors_features, args.with_cuml, num_augmentations=num_augmentations)

        tta_samples_ts_end = timer()

        # saving tta samples creation time
        tta_samples_times = {
            'Gaussian_TTA_Baseline': es_tta_samples_ts_start - gn_tta_samples_ts_start,
            'Euclidean_SMOTE_TTA': ss_tta_samples_ts_start - es_tta_samples_ts_start,
            'Siamese_SMOTE_TTA': ekm_tta_samples_ts_start - ss_tta_samples_ts_start,
            'Euclidean_Kmeans_TTA': skm_tta_samples_ts_start - ekm_tta_samples_ts_start,
            'Siamese_Kmeans_TTA': tta_samples_ts_end - skm_tta_samples_ts_start
        }
        
        # saving tta samples
        algorithms_tta_samples_dict = {
            'Gaussian_TTA_Baseline': gn_tta_samples,
            'Euclidean_SMOTE_TTA': es_tta_samples,
            'Siamese_SMOTE_TTA': ss_tta_samples,
            'Euclidean_Kmeans_TTA': ekm_tta_samples,
            'Siamese_Kmeans_TTA': skm_tta_samples
        }
        

        # making prediction (with the anomaly detection estimator) for every tta sample
        algorithms_tta_predictions_dict = {}

        for algorithm, tta_samples in algorithms_tta_samples_dict.items():
            ae_tta_pred_ts_start = timer()
            ae_tta_pred = ae_test_step_func(tta_samples, trained_ae, loss_func).numpy()

            # if_tta_pred_ts_start = timer()
            # if_tta_pred = if_test_step(tta_samples, trained_if)

            ocs_tta_pred_ts_start = timer()
            ocs_tta_pred = ocs_test_step(tta_samples, trained_ocs)

            lof_tta_pred_ts_start = timer()
            lof_tta_pred = lof_test_step(tta_samples, trained_lof)

            tta_pred_ts_end = timer()
            

            if 'Euclidean' in algorithm:
                neighbors_time = eaclidean_nn_total_time
            elif 'Siamese' in algorithm:
                neighbors_time = siamese_nn_total_time
            else:
                neighbors_time = 0
            # saving tta preds times
            # considering:
            # 1) original test batch inference time
            # 2) finding neighbors
            # 3) test-time augmentations creation
            # 4) augmentations inference time
            times_dict[algorithm] = {
                'Autoencoder': ocs_tta_pred_ts_start - ae_tta_pred_ts_start + tta_samples_times[algorithm] + neighbors_time + times_dict['WO_TTA_Baseline']['Autoencoder'],
                # 'Isolation Forest': ocs_tta_pred_ts_start - if_tta_pred_ts_start + tta_samples_times[algorithm] + neighbors_time + times_dict['WO_TTA_Baseline']['Isolation Forest'],
                'One-Class SVM': lof_tta_pred_ts_start - ocs_tta_pred_ts_start + tta_samples_times[algorithm] + neighbors_time + times_dict['WO_TTA_Baseline']['One-Class SVM'],
                'Local Outlier Factor': tta_pred_ts_end - lof_tta_pred_ts_start + tta_samples_times[algorithm] + neighbors_time + times_dict['WO_TTA_Baseline']['Local Outlier Factor']
            }

            # saving tta preds
            algorithms_tta_predictions_dict[algorithm] = {
                'Autoencoder': ae_tta_pred,
                # 'Isolation Forest': if_tta_pred,
                'One-Class SVM': ocs_tta_pred,
                'Local Outlier Factor': lof_tta_pred
            }


        # merging given test sample's prediction with its tta predictions
        for algorithm, estimators_predictions in algorithms_tta_predictions_dict.items():
            for estimator, tta_preds in estimators_predictions.items():
                wo_tta_pred = algorithms_test_loss['WO_TTA_Baseline'][estimator][step]
                # combine original test samples' predictions with the kmeans regular-NN TTA samples' prediction
                for wo_tta_single_pred, tta_single_pred in list(zip(wo_tta_pred, tta_preds)):
                    combined_tta_loss = np.concatenate([[wo_tta_single_pred], tta_single_pred])
                    algorithms_test_loss[algorithm][estimator].append(np.mean(combined_tta_loss))
    


    # flatten w/o tta baseline test loss and the test_labels vectors
    # algorithms_test_loss['WO_TTA_Baseline'] = np.concatenate(algorithms_test_loss['WO_TTA_Baseline'], axis=0)
    for estimator, estimator_test_loss in algorithms_test_loss['WO_TTA_Baseline'].items():
        algorithms_test_loss['WO_TTA_Baseline'][estimator] = np.concatenate(estimator_test_loss, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    y_true = np.asarray(test_labels).astype(int)


    # calculating AUC
    for algorithm, estimator_final_preds in algorithms_metrics.items():
        for estimator in estimator_final_preds.keys():
            algorithms_metrics[algorithm][estimator] = (roc_auc_score(y_true, algorithms_test_loss[algorithm][estimator]))

    # printing times
    print(times_dict)

    return algorithms_metrics


def ae_test_step():
    # @tf.function
    def test_one_step(inputs, ae, loss_func):
        encoder, decoder = ae[0], ae[1]
        latent_var = encoder(inputs)
        reconstructed = decoder(latent_var)
        reconstruction_loss = loss_func(inputs, reconstructed)

        return reconstruction_loss
    return test_one_step

# def if_test_step(test_X, trained_if):
#     """
#     A test phase with Isolation Forest on the given test set
#     Parameters
#     ----------
#     test_X: numpy ndarray of shape (batch_size, num_augmentations, dataset's features dim) or (batch_size, dataset's features dim). The batch test set
#     trained_if: scikit-learn's IsolationForest. The trained Isolation Forest as anomaly detection estimator
#     """

#     if len(test_X.shape) == 3:
#         batch_anomaly_score = []
#         for one_test_tta_samples in test_X:
#             anomaly_score = -1 * trained_if.score_samples(one_test_tta_samples)
#             batch_anomaly_score.append(anomaly_score)

#         anomaly_score = np.array(batch_anomaly_score)
#     else:
#         anomaly_score = -1 * trained_if.score_samples(test_X)

#     return anomaly_score

def lof_test_step(test_X, trained_lof):
    """
    A test phase with Local Outlier Factor on the given test set
    Parameters
    ----------
    test_X: numpy ndarray of shape (batch_size, dataset's features dim). The batch test set
    trained_lof: scikit-learn's LocalOutlierFactor. The trained Local Outlier Factor as anomaly detection estimator
    """

    if len(test_X.shape) == 3:
        batch_anomaly_score = []
        for one_test_tta_samples in test_X:
            anomaly_score = -1 * trained_lof.score_samples(one_test_tta_samples)
            batch_anomaly_score.append(anomaly_score)

        anomaly_score = np.array(batch_anomaly_score)
    else:
        anomaly_score = -1 * trained_lof.score_samples(test_X)

    return anomaly_score

def ocs_test_step(test_X, trained_ocs):
    """
    A test phase with One-Class SVM on the given test set
    Parameters
    ----------
    test_X: numpy ndarray of shape (batch_size, dataset's features dim). The batch test set
    trained_ocs: scikit-learn's OneClassSVM. The trained One-Class SVM as anomaly detection estimator
    """

    if len(test_X.shape) == 3:
        batch_anomaly_score = []
        for one_test_tta_samples in test_X:
            anomaly_score = -1 * trained_ocs.score_samples(one_test_tta_samples)
            batch_anomaly_score.append(anomaly_score)

        anomaly_score = np.array(batch_anomaly_score)
    else:
        anomaly_score = -1 * trained_ocs.score_samples(test_X)

    return anomaly_score

def generate_random_noise_tta_samples(x_batch_test, num_augmentations):
    """
    Generating TTA with random Gaussian noise
    Parameters
    ----------
    x_batch_test: ndarray of shape (batch_size, #features). The features of each test sample in the batch
    num_augmentations: int. The nubmer of augmentations to produce 
    """

    # scale = 0.2
    random_noise = np.random.normal(size=(x_batch_test.shape[0], num_augmentations, x_batch_test.shape[1]))
    # adding the noise to the original batch test samples. expanding the middle dim of x_batch_test to make it (batch_size, 1, dataset_features_dim)
    gaussian_tta_samples = np.expand_dims(x_batch_test, axis=1) + random_noise
    
    return gaussian_tta_samples

def generate_kmeans_tta_samples(batch_neighbors_features, with_cuML, num_augmentations):
    """
    Generating TTA with trained k-means
    Parameters
    ----------
    batch_neighbors_features: numpy ndarray of shape (batch_size, num_neighbors, #features). The features of each neighbor of each test sample that is in the batch
    with_cuML: bool. If True, then using cuML's k-Means model otherwise using scikit-learn's k-Means model
    num_augmentations: int. The number of augmentations to produce
    """

    batch_tta_samples = []
    for neighbors_features in batch_neighbors_features:
        if with_cuML:
            kmeans_model = cuKMeans(n_clusters=num_augmentations, random_state=1234)
        else:
            kmeans_model = cuKMeans(n_clusters=num_augmentations, random_state=1234)
        neighbors_features = neighbors_features.astype(np.float32)
        kmeans_model.fit(X=neighbors_features)
        tta_samples = kmeans_model.cluster_centers_
        # appending to the batch tta samples
        batch_tta_samples.append(tta_samples)
    
    return np.array(batch_tta_samples)

def generate_oversampling_tta_samples(oversampling_batch_neighbors_features, num_neighbors, num_augmentations, oversampling_method):
    """
    Generating TTA with oversampling method (SMOTE)
    Parameters
    ----------
    oversampling_batch_neighbors_features: numpy ndarray of shape (batch_size, num_neighbors, #features). The features of each neighbor of each test sample that is in the batch
    num_neighbors: int. The number of neighbor each test sample in the batch has
    num_augmentations: int. The number of augmentations to produce
    oversampling_meethod: function. SMOTE function
    """

    batch_size, features_dim = oversampling_batch_neighbors_features.shape[0], oversampling_batch_neighbors_features.shape[-1]

    oversampling_batch_tta_samples = np.zeros((batch_size, num_augmentations, features_dim))
    for index_in_batch, original_neighbors_features in enumerate(oversampling_batch_neighbors_features):
        original_neighbors_labels = np.zeros((original_neighbors_features.shape[0],))
        
        # create fake samples for the imblearn dataset
        fake_neighbors_features = np.zeros((num_neighbors + num_augmentations, features_dim))
        fake_neighbors_labels = np.ones((fake_neighbors_features.shape[0],))

        # create the imblearn dataset
        imblearn_features = np.concatenate([original_neighbors_features, fake_neighbors_features])
        imblearn_labels = np.concatenate([original_neighbors_labels, fake_neighbors_labels])

        oversampling_obj = oversampling_method(k_neighbors=num_neighbors-1, random_state=42)
        X_res, y_res = oversampling_obj.fit_resample(imblearn_features, imblearn_labels)

        current_augmentations = X_res[-num_augmentations:]
        oversampling_batch_tta_samples[index_in_batch] = current_augmentations
    
    return oversampling_batch_tta_samples

def print_test_results(algorithms_folds_metrics, args):
    """
    Printing the results metrics of all of the evaluated algorithms
    Parameters
    ----------
    algorithms_folds_metrics: dictionaty. A dictionary that holds the results metrics for every algorithm in every k-fold split
    args: argparse args. The args given to the program
    """
    
    max_col_len=len(max(evaluated_estimators, key=len))

    print(f"--- AUC on {args.n_folds}-fold cross-validation with {args.num_neighbors} neighbors and {args.num_augmentations} augmentations ---")

    # creating table structure
    table = PrettyTable()
    # table.field_names = ['Algorithm', 'Autoencoder', 'Isolation Forest', 'One-Class SVM', 'Local Outlier Factor']
    table.field_names = ['Algorithm', 'Autoencoder', 'One-Class SVM', 'Local Outlier Factor']

    for algorithm, metrics in algorithms_folds_metrics.items():
        ae_mean_std_auc = "{:0.3f}+-{:0.3f}".format(metrics['Autoencoder'].mean(axis=0), metrics['Autoencoder'].std(axis=0))
        # if_mean_std_auc = "{:0.3f}+-{:0.3f}".format(metrics['Isolation Forest'].mean(axis=0), metrics['Isolation Forest'].std(axis=0))
        ocs_mean_std_auc = "{:0.3f}+-{:0.3f}".format(metrics['One-Class SVM'].mean(axis=0), metrics['One-Class SVM'].std(axis=0))
        lof_mean_std_auc = "{:0.3f}+-{:0.3f}".format(metrics['Local Outlier Factor'].mean(axis=0), metrics['Local Outlier Factor'].std(axis=0))
        # table.add_row([algorithm, ae_mean_std_auc, if_mean_std_auc, ocs_mean_std_auc, lof_mean_std_auc])
        table.add_row([algorithm, ae_mean_std_auc, ocs_mean_std_auc, lof_mean_std_auc])
    
    print(table)