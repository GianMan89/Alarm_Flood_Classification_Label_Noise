import numpy as np
from .CASIM_arsenal import Arsenal


class CASIM:
    """
    CASIM classifier
    """

    def __init__(
        self,
        params={
            "num_features": 672,
            "n_estimators": 25,
            "n_jobs_multirocket": 1,
            "random_state": 42,
            "alphas": np.logspace(-3, 3, 10),
        },
    ):
        self.clf = None
        self.num_features = params["num_features"]
        self.n_estimators_param = params["n_estimators"]
        self.n_jobs_multirocket = params["n_jobs_multirocket"]
        self.random_state = params["random_state"]
        self.alphas = params["alphas"]

        self.X_length = None

    @property
    def __name__(self):
        return "CASIM"

    def fit(self, X, y):
        # train Arsenal ensemble of MultiRocket classifiers
        self.clf = Arsenal(
            num_features=self.num_features,
            n_jobs_multirocket=self.n_jobs_multirocket,
            n_estimators=self.n_estimators_param,
            random_state=self.random_state,
            alphas=self.alphas,
        )
        self.clf.fit(X, y)
        # save the length of the input data
        self.X_length = X.shape[2]

    def predict_proba(self, X):
        # if the length of the input data is different from the length of the
        # training data we need to zero-pad the input data
        if X.shape[2] != self.X_length:
            # get the difference in length
            diff = self.X_length - X.shape[2]
            # zero-pad the input data
            X = np.pad(X, ((0, 0), (0, 0), (0, diff)), "constant")
        # get the posterior class probabilities
        y_scores = self.clf._predict_proba(X)
        return np.array(y_scores)
