from ._euclidean import EuclideanNearestNeighbors

def data_selector_mapper(data_selector_distance_metric_str, **kwargs):
    if data_selector_distance_metric_str == "euclidean":
        return EuclideanNearestNeighbors(**kwargs)
    else:
        raise ValueError("Invalid data selector distance metric!")