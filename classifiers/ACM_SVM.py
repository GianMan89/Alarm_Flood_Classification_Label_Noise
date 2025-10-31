import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ACM_SVM:
    """
    Alarm Coactivation Matrix Support Vector Machine classifier
    """

    def __init__(self, params={}):
        self.clf = None

    @property
    def __name__(self):
        return "ACM_SVM"

    def fit(self, X, y):
        # get alarm coactivations
        X_acm = self.calc_coactivation(X)
        # train SVM with one-vs-one scheme
        self.clf = make_pipeline(StandardScaler(), SVC(probability=True))
        self.clf.fit(X_acm, y)

    def predict_proba(self, X):
        # get alarm coactivations
        X_acm = self.calc_coactivation(X)
        # get the posterior class probabilities
        y_scores = self.clf.predict_proba(X_acm)
        return np.array(y_scores)

    def calc_coactivation(self, X):
        """
        Compute the coactivation matrix for each sample in the 3D numpy array X.
        Each coactivation matrix contains Jaccard scores for each pair of variables.

        :param X: 3D numpy array of shape (n_samples, n_variables, n_timesteps)
        :return: 2D numpy array of shape (n_samples, n_variables*(n_variables-1)/2)
        """
        # Ensure binary activations exactly as in original code (value == 1)
        A = (X == 1).astype(np.float64)  # shape: (S, V, T)
        n_samples, n_variables, _ = A.shape

        # Intersection counts for all variable pairs per sample:
        # intersections[s, i, j] = sum_t A[s, i, t] * A[s, j, t]
        intersections = np.einsum('svt,swt->svw', A, A, optimize=True)

        # Row sums per variable (activations per variable over time)
        row_sums = A.sum(axis=2)  # shape: (S, V)

        # Union counts using |A ∪ B| = |A| + |B| - |A ∩ B|
        unions = row_sums[:, :, None] + row_sums[:, None, :] - intersections  # (S, V, V)

        # Jaccard with union==0 -> 0 (as in the original)
        with np.errstate(divide='ignore', invalid='ignore'):
            jaccard = np.where(unions > 0, intersections / unions, 0.0)  # (S, V, V)

        # Extract upper-triangular (v1 < v2) and flatten per sample
        iu = np.triu_indices(n_variables, k=1)
        result = jaccard[:, iu[0], iu[1]]  # shape: (S, V*(V-1)//2)

        return result