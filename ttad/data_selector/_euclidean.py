from ._base import DataSelectorBase

class EuclideanNearestNeighbors(DataSelectorBase):
    def __init__(self,
                 use_GPU=False,
                 **kwargs) -> None:
        kwargs['metric'] = 'euclidean'
        super().__init__(use_GPU, **kwargs)


    def fit(self, X):
        self.inner_model.fit(X)


    def get_neighbors(self, X, n_neighbors):
        return self.inner_model(X=X, n_neighbors=n_neighbors, return_distance=False)