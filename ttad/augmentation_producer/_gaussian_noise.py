import numpy as np
from ._base import AugmentationProducerInterface

class GaussianNoiseAP(AugmentationProducerInterface):

    def generate_tta(X, neighbors, num_augmentations, use_GPU):
        """
        Generating TTA with random Gaussian noise.
        This implementation does not consider the neighbors
        
        """

        # scale = 0.2
        random_noise = np.random.normal(size=(X.shape[0], num_augmentations, X.shape[1]))
        # adding the noise to the each test samples. expanding the middle dim of X to make it (num_samples, 1, #features)
        gaussian_tta_samples = np.expand_dims(X, axis=1) + random_noise
        
        return gaussian_tta_samples