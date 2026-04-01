import numpy as np

def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    data = np.array(data)
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    normalized_data = (data - data_min) / (data_max - data_min + 1e-15)
    return normalized_data.tolist()
