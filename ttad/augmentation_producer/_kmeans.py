import numpy as np
from cuml.cluster import KMeans as cuKMeans
from sklearn.cluster import KMeans

from ._base import AugmentationProducerInterface

class KMeansAP(AugmentationProducerInterface):

    def generate_tta(X, neighbors, num_augmentations, use_GPU):
        """
        Generating TTA using a k-Means centroids
        
        """

        kmeans_tta_samples = []
        for neighbors_features in neighbors:
            if use_GPU:
                kmeans_model = cuKMeans(n_clusters=num_augmentations, random_state=1234)
            else:
                kmeans_model = KMeans(n_clusters=num_augmentations, random_state=1234)

            neighbors_features = neighbors_features.astype(np.float32)
            
            kmeans_model.fit(X=neighbors_features)
            tta_samples = kmeans_model.cluster_centers_
            # appending to the tta samples
            kmeans_tta_samples.append(tta_samples)
        
        return np.array(kmeans_tta_samples)