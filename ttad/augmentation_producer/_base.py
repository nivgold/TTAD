from abc import ABCMeta, abstractmethod

class AugmentationProducerInterface(metaclass=ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, 'generate_tta'),
            callable(subclass.generate_tta)
        )


    @staticmethod
    @abstractmethod
    def generate_tta(X, neighbors, num_augmentations, use_GPU):
        """
        Generates TTAs given the data `X_with_neighbors`.

        Parameters
        ----------
        X: numpy ndarray of shape (num_samples, #features). The test set
        neighbors: numpy ndarray of shape (num_samples, num_neighbors, #features). The features of each neighbor of each test sample that is in the test set
        num_augmentations: int. The number of augmentations to produce
        use_GPU: bool. If True, then using GPU if needed
        """
        raise NotImplementedError
