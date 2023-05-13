from ..utils import is_GPU_available
from abc import ABC, abstractmethod

from ..augmentation_producer import augmentation_producer_mapper
from ..data_selector import data_selector_mapper

from ..data_selector._base import DataSelectorBase
from ..augmentation_producer._base import AugmentationProducerInterface


class TTADWrapperBase(ABC):
    def __init__(self,
                 model,
                 num_augmentations,
                 num_neighbors,
                 data_selector="euclidean",
                 augmentation_producer="kmeans",
                 use_GPU=False) -> None:
        
        super().__init__()


        if use_GPU:
            if is_GPU_available() == False:
                raise ValueError("GPU is not available!")
        self.use_GPU = use_GPU

        self.trained_estimator = model
        self.num_augmentations = num_augmentations
        self.num_neighbors = num_neighbors

        if isinstance(data_selector, str):
            data_selector = data_selector_mapper(data_selector, use_GPU)
        elif not issubclass(data_selector, DataSelectorBase):
            raise ValueError("Invalid Data Selector instance")
        self.data_selector = data_selector

        if isinstance(augmentation_producer, str):
            augmentation_producer = augmentation_producer_mapper(augmentation_producer)
        elif not issubclass(augmentation_producer, AugmentationProducerInterface):
            raise ValueError("Invalid Augmentation Producer instance")
        self.augmentation_producer = augmentation_producer
        

    @abstractmethod
    def fit(self, X):
        """
        Fitting the given model as well as TTAD's essential models and params.
        """
        raise NotImplementedError


    @abstractmethod
    def predict(self, X):
        """
        Inference using given model, while performing a TTA-based test.
        """
        raise NotImplementedError