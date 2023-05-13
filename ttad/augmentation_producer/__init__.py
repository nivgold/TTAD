from ._gaussian_noise import GaussianNoiseAP
from ._smote import SMOTEAP
from ._kmeans import KMeansAP

def augmentation_producer_mapper(augmentation_producer_str, **kwargs):
    if augmentation_producer_str == "gaussian_noise":
        return GaussianNoiseAP
    elif augmentation_producer_str == "smote":
        return SMOTEAP
    elif augmentation_producer_str == "kmeans":
        return KMeansAP
    else:
        raise ValueError("Not a Valid Augmentation Producer")