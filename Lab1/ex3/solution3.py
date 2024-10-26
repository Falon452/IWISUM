from sklearn.svm import OneClassSVM
import numpy as np

def detect(train_data: np.ndarray, test_data: np.ndarray) -> list:
    oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.007)
    oc_svm.fit(train_data)
    anomalies_oc_svm = oc_svm.predict(test_data)
    anomalies_oc_svm = np.where(anomalies_oc_svm == -1, 1, 0)
    return anomalies_oc_svm.tolist()
