import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

SEED = 1

def detect_cov(data: np.ndarray, outliers_fraction: float) -> list:
    envelope = EllipticEnvelope(contamination=outliers_fraction)
    envelope.fit(data)
    anomalies = envelope.predict(data)
    return np.where(anomalies == -1, 1, 0).tolist()

def detect_ocsvm(data: np.ndarray, outliers_fraction: float) -> list:
    oc_svm = OneClassSVM(kernel='rbf', nu=outliers_fraction)
    oc_svm.fit(data)
    anomalies = oc_svm.predict(data)
    return np.where(anomalies == -1, 1, 0).tolist()

def detect_iforest(data: np.ndarray, outliers_fraction: float) -> list:
    isolation_forest = IsolationForest(contamination=outliers_fraction, random_state=SEED)
    isolation_forest.fit(data)
    anomalies = isolation_forest.predict(data)
    return np.where(anomalies == -1, 1, 0).tolist()

def detect_lof(data: np.ndarray, outliers_fraction: float) -> list:
    lof = LocalOutlierFactor(n_neighbors=520, contamination=outliers_fraction)
    anomalies = lof.fit_predict(data)
    return np.where(anomalies == -1, 1, 0).tolist()
