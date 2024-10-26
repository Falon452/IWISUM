import numpy as np


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    mu = np.mean(train_data)
    sigma = np.std(train_data)
    lower_bound = mu - 3.6 * sigma
    upper_bound = mu + 2.6 * sigma
    predictions = np.where((test_data < lower_bound) | (test_data > upper_bound), 1, 0)
    return predictions.tolist()
