import numpy as np

from ._base import TTADWrapperBase


class TTADRegressor(TTADWrapperBase):
    def __init__(self,
                 model,
                 num_augmentations,
                 num_neighbors,
                 data_selector_distance_metric="euclidean",
                 augmentation_producer="kmeans",
                 use_GPU=False) -> None:
        
        super().__init__(model,
                         num_augmentations,
                         num_neighbors,
                         data_selector_distance_metric,
                         augmentation_producer,
                         use_GPU)


    def fit(self, X):
        # training the given model
        self.trained_estimator.fit(X)

        # save training set
        self.training_set = X.copy()


    def predict(self, X):
        # (#train_samples + #test_samples, #features)
        X_data_selector = np.concatenate(self.training_set, X, axis=0)

        # training the data selector
        self.data_selector.fit(X_data_selector)

        # (#test_samples, )
        original_y_pred = self.trained_estimator.fit(X)

        # use data selector component
        # (#test_samples, num_neighbors)
        neighbors_idx = self.data_selector.get_neighbors(X, self.num_neighbors)
        # (#test_samples, num_neighbors, #features)
        neighbors = X_data_selector[neighbors_idx]

        # produce TTA
        # (#test_samples, num_augmentations, #features)
        tta_samples = self.augmentation_producer.generate_tta(X, neighbors, self.num_augmentations, self.use_GPU)

        # inference for each test sample's TTAs
        tta_samples_pred = []
        for augmentations_features in tta_samples:
            augmentations_pred = self.trained_estimator.predict(augmentations_features)
            tta_samples_pred.append(augmentations_pred)
        tta_samples_pred = np.array(tta_samples_pred)

        # ensembling the TTAs predictions with the original test sample prediction
        y_pred = np.column_stack([tta_samples_pred, original_y_pred])        
        y_pred = y_pred.mean(axis=-1)

        return y_pred
