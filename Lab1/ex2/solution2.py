import numpy as np
from sklearn.covariance import MinCovDet


def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    mcd = MinCovDet().fit(train_data)
    distances_train = mcd.mahalanobis(train_data)
    max_distance = np.max(distances_train)
    distances_test = mcd.mahalanobis(test_data)
    artifacts = np.where(distances_test > max_distance, 1, 0)
    return artifacts.tolist()
