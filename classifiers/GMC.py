import numpy as np


class GMC:
    """
    Guess Most Common class classifier
    """

    def __init__(self, params={}):
        self.most_frequent_class = None
        self.n_classes = None

    def fit(self, _, y_train):
        self.most_frequent_class = np.bincount(y_train).argmax()
        self.n_classes = len(np.unique(y_train))

    def predict_proba(self, X):
        y_predict = []
        for i in X:
            # predict the most frequent class
            # add a list of probabilities for each class
            # for the most frequent class, the probability is 1
            # for all other classes, the probability is 0
            y_predict.append(
                [
                    1 if i == self.most_frequent_class else 0
                    for i in range(self.n_classes)
                ]
            )
        return np.array(y_predict)
