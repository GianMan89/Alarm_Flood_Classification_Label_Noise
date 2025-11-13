import math
import numpy as np
import classifiers.utils as utils

class EAC_KNN:
    """
    Exponentially Attenuation Coefficient K-Nearest Neighbor classifier
    """

    def __init__(self, params={"attenuation_coefficient_per_min": 0.0667, "n_neighbors": 3}):
        # convert attenuation coefficient from min to hour
        self.attenuation_coefficient = (
            params["attenuation_coefficient_per_min"] * 60
        )
        self.k = params.get("n_neighbors", 3)
        self.X_train_features = None
        self.y_train = None
        self.classes = None

    @property
    def __name__(self):
        return "EAC_KNN"

    def fit(self, X, y):
        self.n_alm_vars = X.shape[1]
        self.y_train = y
        self.classes = np.unique(y)
        # Cache converted alarms
        X_converted = utils.convert_alarms(X)
        self.X_train_features = self.get_feature_vectors(X_converted)

    def predict_proba(self, X):
        # Cache converted alarms
        X_converted = utils.convert_alarms(X)
        X_test_features = self.get_feature_vectors(X_converted)
        return np.array([self.get_k_nearest_neighbors(x) for x in X_test_features])
    
    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = [self.classes[np.argmax(y_proba[i])] for i in range(y_proba.shape[0])]
        return np.array(y_pred)

    def calculate_distances(self, reference_vector, vector_set):
        distances = np.linalg.norm(vector_set - reference_vector, axis=1)
        return distances

    def get_k_nearest_neighbors(self, x):
        # Calculate distances to all training samples
        distances = self.calculate_distances(x, self.X_train_features)
        
        # Get indices of k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:self.k]
        
        # Get labels of k nearest neighbors
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        # Calculate class probabilities based on k nearest neighbors
        class_probs = []
        for class_label in self.classes:
            count = np.sum(k_nearest_labels == class_label)
            class_probs.append(count / self.k)
        
        return class_probs

    def get_feature_vectors(self, X):
        X_alarms = self.get_alarm_flood_vector(X)
        X_times = self.get_time_vector(X)

        X_alarms = np.array(X_alarms)
        X_times = np.array(X_times)
        
        # Vectorized computation of exponential attenuation
        X_vectors = X_alarms * np.exp(-self.attenuation_coefficient * X_times)
        return X_vectors

    def get_alarm_flood_vector(self, X):
        # Vectorized method to create alarm flood vectors
        X_alarms = np.zeros((len(X), self.n_alm_vars))
        for i, x in enumerate(X):
            alarm_types = [alm.type for alm in x]
            X_alarms[i, alarm_types] = 1
        return X_alarms

    def get_time_vector(self, X):
        # Vectorized method to create time vectors
        X_times = np.full((len(X), self.n_alm_vars), np.inf)
        for i, alarms in enumerate(X):
            for alarm in alarms:
                if X_times[i, alarm.type] == np.inf:  # If not already set
                    X_times[i, alarm.type] = alarm.start
        # Replace any remaining inf with 0 (no time set)
        X_times[X_times == np.inf] = 0
        return X_times
