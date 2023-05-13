from abc import ABC, abstractmethod

from sklearn.neighbors import NearestNeighbors
from cuml.neighbors import NearestNeighbors as cuNearestNeighbors


class DataSelectorBase(ABC):
    def __init__(self,
                 use_GPU=False,
                 **kwargs) -> None:
        super().__init__()

        if use_GPU:
            self.inner_model = cuNearestNeighbors(**kwargs)
        else:
            self.inner_model = NearestNeighbors(**kwargs)


    @abstractmethod
    def fit(self, X):
        """
        """
        raise NotImplementedError


    @abstractmethod
    def get_neighbors(self, X, n_neighbors):
        """
        """
        raise NotImplementedError