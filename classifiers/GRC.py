import numpy as np


class GRC:
    """
    Guess Random Class classifier
    """

    def __init__(self, params={}):
        self.class_labels = None
        self.n_classes = None

    def fit(self, _, y_train):
        self.class_labels = np.unique(y_train)
        self.n_classes = len(self.class_labels)

    def predict_proba(self, X):
        y_predict = []
        for i in X:
            # predict a random class
            # add a list of probabilities for each class
            # for the selected random class, the probability is 1
            # for all other classes, the probability is 0
            y_predict.append(
                [
                    1 if i == np.random.choice(self.class_labels, 1) else 0
                    for i in range(self.n_classes)
                ]
            )
        return np.array(y_predict)
    
    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = [np.argmax(y_proba[i]) for i in range(y_proba.shape[0])]
        return np.array(y_pred)
