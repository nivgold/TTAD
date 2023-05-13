import numpy as np
from imblearn.over_sampling import SMOTE

from ._base import AugmentationProducerInterface

class SMOTEAP(AugmentationProducerInterface):

    def generate_tta(X, neighbors, num_augmentations, use_GPU):
        """
        Generating TTA with the oversampling SMOTE method
        
        """

        num_samples, num_neighbors, features_dim = neighbors.shape[0], neighbors.shape[1], neighbors.shape[2]

        oversampling_tta_samples = np.zeros((num_samples, num_augmentations, features_dim))
        for index, original_neighbors_features in enumerate(neighbors):
            original_neighbors_labels = np.zeros((original_neighbors_features.shape[0],))
            
            # create fake samples for the imblearn dataset
            fake_neighbors_features = np.zeros((num_neighbors + num_augmentations, features_dim))
            fake_neighbors_labels = np.ones((fake_neighbors_features.shape[0],))

            # create the imblearn dataset
            imblearn_features = np.concatenate([original_neighbors_features, fake_neighbors_features])
            imblearn_labels = np.concatenate([original_neighbors_labels, fake_neighbors_labels])

            oversampling_obj = SMOTE(k_neighbors=num_neighbors-1, random_state=42)
            X_res, y_res = oversampling_obj.fit_resample(imblearn_features, imblearn_labels)

            current_augmentations = X_res[-num_augmentations:]
            oversampling_tta_samples[index] = current_augmentations
        
        return oversampling_tta_samples