import numpy as np
import tensorflow as tf

from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
from cuml.cluster import KMeans as cuKMeans

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

from autoencoder_model import Encoder, Decoder
from siamese_model import SiameseNetwork


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def train(X, y, siamese_pairs, folded_train_datasets_list, features_dim, args):
    """
    Training the framework. Training anomaly detector model (AE), training NN model, training Siamese neural network.
    Parameters
    ----------
    X. numpy ndarray of shape (#num_samples, #features). The dataset's features
    y. numpy ndarray of shape (#num_samples, ). The dataset's labels
    siamese_pairs. tuple. A tuple containing both siamese pairs features and labels for training the Siamese network
    folded_train_datasets_list. list of TF's Dataset. The training sets of every k-fold split
    features_dim. int. The dimensionality of the dataset
    args. argparse args. The args given to the program
    """

    trained_siamese_network = train_siamese_model(siamese_pairs, args.siamese_batch_size, args.siamese_n_epochs)
    euclidean_nn_model, siamese_nn_model = train_nn_model(X, trained_siamese_network, args.with_cuml)

    trained_estimators_list = train_estimator(folded_train_datasets_list, features_dim, args)

    return trained_estimators_list, trained_siamese_network, euclidean_nn_model, siamese_nn_model

def train_siamese_model(siamese_data, batch_size, n_epochs):
    """
    Training Siamese network
    Parameters
    ----------
    siamese_data: tuple. A tuple containing both siamese pairs features and labels for training the Siamese network
    batch_size: int. The batch size used to train the Siamese network
    n_epochs: int. The nubmer of epochs used for training the Siamese network
    """

    pairs_X, pairs_y = siamese_data
    
    # defining Siamese network
    siamese_network = SiameseNetwork()
    # compile the Siamese netowrk using Adam optimizer and binary cross-entropy as loss function
    siamese_network.compile(optimizer='adam', loss='binary_crossentropy')

    # training the Siamese network
    siamese_network.fit(
        [pairs_X[0], pairs_X[1]],
        pairs_y,
        batch_size=batch_size,
        epochs=n_epochs,
        verbose=0
    )

    # return only the internal model because with this part we are making use later
    return siamese_network.internal_model

def train_nn_model(nn_data, trained_siamese_network, with_cuml):
    """
    Training eulidean distance-based NN model and a Siamese distance-based NN model
    Parameters
    ----------
    nn_data. pandas DataFrame. The dataset's features
    """

    if with_cuml:
        print("--- Using cuML ---")
        # defining cuML euclidean Nearest Neighbors model
        euclidean_nn_model = cuNearestNeighbors()
        # defining cuML siamese Nearest Neighbors model
        siamese_nn_model = cuNearestNeighbors(metric='cosine')
    else:
        print("--- Not using cuML ---")
        # defining sklearn euclidean Nearest Neighbors model
        euclidean_nn_model = NearestNeighbors()
        # defining sklearn siamese Nearest Neighbors model
        siamese_nn_model = NearestNeighbors(metric='cosine')
    
    # training euclidean Nearest Neighbors model
    euclidean_nn_model.fit(nn_data)

    # converting NN data to a "Siamese latent space"
    latent_space_data = trained_siamese_network(nn_data).numpy()
    # training siamese Nearest Neighbors model
    siamese_nn_model.fit(latent_space_data)

    return euclidean_nn_model, siamese_nn_model

def train_estimator(folded_train_datasets_list, features_dim, args):
    """
    Training the anomaly detector model (AE) for every k-fold split.
    Parameters
    ----------
    folded_train_datasets_list. list of TF's Dataset. The training sets of every k-fold split
    features_dim. int. The dimensionality of the dataset
    args. argparse args. The args given to the program
    """

    estimators_list = []

    for split_index in range(args.n_folds):
        # train set-up
        train_ds = folded_train_datasets_list[split_index]
        train_step_func = ae_train_step()
        print(f"--- Training K-Fold Split index: {split_index+1} ---")
        # training AE on current k-fold split
        trained_ae = ae_training_loop(train_ds, train_step_func, features_dim, args)
        # training IF on current k-fold split
        trained_if = if_training(train_ds)
        # training OCS on current k-fold split
        trained_ocs = ocs_training(train_ds)
        # training LOF on current k-fold split
        trained_lof = lof_training(train_ds)

        # adding to models list
        estimators_list.append({'ae': trained_ae, 'if': trained_if, 'ocs': trained_ocs, 'lof': trained_lof})
    
    return estimators_list

def ae_training_loop(train_ds, train_step_func, features_dim, args):
    """
    The training loop of an Autoencoder as anomaly detector
    Parameters
    ----------
    train_ds. TF's Dataset. The training dataset
    train_step_func. function. The function that is used for performing single train step
    features_dim. int. The dimensionality of the dataset
    args. argparse args. The args given to the program
    """

    encoder = Encoder(input_shape=features_dim)
    decoder = Decoder(original_dim=features_dim)

    # define loss function
    loss_func = tf.keras.losses.MeanSquaredError()
    # define optimizer
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(1, args.num_epochs+1):
        epoch_loss_mean = tf.keras.metrics.Mean()

        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            loss = train_step_func(x_batch_train, encoder, decoder, optimizer, loss_func)

            # keep track of the metrics
            epoch_loss_mean.update_state(loss)

        # update metrics after each epoch
        if epoch==1 or epoch%100 == 0:
            print(f'Epoch {epoch} loss mean: {epoch_loss_mean.result()}')
    
    return (encoder, decoder)

def ae_train_step():
    @tf.function
    def train_one_step(inputs, encoder, decoder, optimizer, loss_func):
        with tf.GradientTape() as tape:
            latent_var = encoder(inputs)
            outputs = decoder(latent_var)
            loss = loss_func(inputs, outputs)

            trainable_vars = encoder.trainable_variables \
                            + decoder.trainable_variables

        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))

        return loss
    return train_one_step

def if_training(train_ds):
    """
    The training of an Isolation Forest as anomaly detector
    Parameters
    ----------
    train_ds: TF's Dataset. The trainind dataset
    """
    
    # collect the training set
    train_X = []
    for x_batch_train, y_batch_train in train_ds:
        train_X.append(x_batch_train)
    train_X = np.concatenate(train_X, axis=0)

    if_clf = IsolationForest(random_state=42, n_jobs=-1).fit(train_X)

    return if_clf

def lof_training(train_ds):
    """
    The training of an Local Outlier Factor (LOF) as anomaly detector
    Parameters
    ----------
    train_ds: TF's Datase.t The training dataset
    """

    # collect the training set
    train_X = []
    for x_batch_train, y_batch_train in train_ds:
        train_X.append(x_batch_train)
    train_X = np.concatenate(train_X, axis=0)

    lof_clf = LocalOutlierFactor(novelty=True, n_jobs=-1).fit(train_X)

    return lof_clf

def ocs_training(train_ds):
    """
    The training of an one-class SVM as anomaly detector
    Parameters
    ----------
    train_ds: TF's Dataset. The training dataset
    """

    # collect the training set
    train_X = []
    for x_batch_train, y_batch_train in train_ds:
        train_X.append(x_batch_train)
    train_X = np.concatenate(train_X, axis=0)

    ocs_clf = OneClassSVM().fit(train_X)

    return ocs_clf