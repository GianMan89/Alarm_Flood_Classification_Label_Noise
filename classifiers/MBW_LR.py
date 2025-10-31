import math
import numpy as np
from sklearn.linear_model import LogisticRegression

import classifiers.utils as utils


class MBW_LR:
    """
    Modified Bag of Words Logistic Regression classifier
    """

    def __init__(
        self,
        params={
            "penalty": None,
            "fit_intercept": False,
            "solver": "lbfgs",
            "multi_class": "ovr",
            "decision_bounds": False,
            "confidence_interval": 1.96,
        },
    ):
        self.penalty = params["penalty"]
        self.fit_intercept = params["fit_intercept"]
        self.solver = params["solver"]
        self.multi_class = params["multi_class"]
        self.decision_bounds = params["decision_bounds"]
        self.confidence_interval = params["confidence_interval"]

        self.clf = None
        self.class_bounds = {}
    
    @property
    def __name__(self):
        return "MBW_LR"

    def fit(self, X, y):
        self.n_alm_vars = X.shape[1]
        self.idf = np.array([0 for i in range(self.n_alm_vars)])
        X_train_convert = utils.convert_alarms(X)
        X_train_count = self.get_alarm_count(X_train_convert)
        tf_train = self.get_tf(X_train_count)
        time_weights_train = self.get_time_weight(X_train_convert)
        self.idf = self.get_idf(X_train_count)
        mbow_vectors_train = self.get_mbow_vectors(
            tf_train, time_weights_train
        )
        # Check for NaN values and replace with 0
        if np.isnan(mbow_vectors_train).any():
            mbow_vectors_train = np.nan_to_num(mbow_vectors_train)
        self.clf = LogisticRegression(
            penalty=self.penalty,
            fit_intercept=self.fit_intercept,
            solver=self.solver,
            multi_class=self.multi_class,
        ).fit(mbow_vectors_train, y)
        # calculate class boundaries
        if self.decision_bounds == True:
            y_scores = self.clf.predict_proba(mbow_vectors_train)
            for i in range(len(self.clf.classes_)):
                cl = self.clf.classes_[i]
                probs_1 = y_scores[np.where([y == cl])[1]][:, i]
                probs_2 = 1 + (1 - probs_1)
                probs = np.concatenate((probs_1, probs_2), axis=0)
                self.class_bounds[cl] = max(
                    [0.5, self.confidence_interval * np.std(probs)]
                )

    def predict_proba(self, X):
        # generate modified bag of words vectors for X_test
        X_test_convert = utils.convert_alarms(X)
        X_test_count = self.get_alarm_count(X_test_convert)
        tf_test = self.get_tf(X_test_count)
        time_weights_test = self.get_time_weight(X_test_convert)
        mbow_vectors_test = self.get_mbow_vectors(tf_test, time_weights_test)
        # Check for NaN values and replace with 0
        if np.isnan(mbow_vectors_test).any():
            mbow_vectors_test = np.nan_to_num(mbow_vectors_test)
        # get the posterior class probabilities
        y_scores = self.clf.predict_proba(mbow_vectors_test)
        return y_scores

    def get_alarm_count(self, X):
        X_count = []
        for i in range(len(X)):
            X_count_i = [0 for m in range(self.n_alm_vars)]
            for j in range(len(X[i])):
                X_count_i[X[i][j].type] += 1
            X_count.append(X_count_i)
        return X_count

    def get_tf(self, X):
        tf = []
        for i in range(len(X)):
            sum_alarms = sum(X[i])
            tf.append(list(np.array(X[i]) / sum_alarms))
        return tf

    def get_idf(self, X):
        idf = [0 for i in range(self.n_alm_vars)]
        n_floods = len(X)
        for j in range(self.n_alm_vars):
            for i in range(len(X)):
                if X[i][j] > 0:
                    idf[j] += 1
        for j in range(self.n_alm_vars):
            if idf[j] > 0:
                idf[j] = math.log(n_floods / idf[j])
        return idf

    def get_time_weight(self, X):
        time_weights = []
        t_max = self.get_t_max(X)
        t_a = self.get_t_a(X)
        for i in range(len(X)):
            time_weights_i = [0 for m in range(self.n_alm_vars)]
            for j in range(self.n_alm_vars):
                if t_a[i][j] > 0:
                    time_weights_i[j] = math.log(t_max[i] / t_a[i][j])
            time_weights.append(time_weights_i)
        return time_weights

    def get_t_max(self, X):
        t_max = []
        for i in range(len(X)):
            if len(X[i]) == 0:
                t_max.append(1)
            else:
                t_max.append(X[i][-1].start)
        return t_max

    def get_t_a(self, X):
        t_a = []
        for i in range(len(X)):
            t_a_i = [0 for m in range(self.n_alm_vars)]
            checked_types = []
            for j in range(len(X[i])):
                if X[i][j].type not in checked_types:
                    t_a_i[X[i][j].type] = X[i][j].start
                    checked_types.append(X[i][j].type)
            t_a.append(t_a_i)
        return t_a

    def get_mbow_vectors(self, tf, time_weights):
        mbow_vectors = []
        for i in range(len(tf)):
            mbow_vector_i = [0 for m in range(self.n_alm_vars)]
            for j in range(self.n_alm_vars):
                mbow_vector_i[j] = tf[i][j] * time_weights[i][j] * self.idf[j]
            mbow_vectors.append(mbow_vector_i)
        return np.array(mbow_vectors)
