import numpy as np
import pandas as pd


class WDI_1NN:
    def __init__(self, params={"template_threshold": 0.5}):
        self.classes = None
        self.templates = None
        self.weights_novelty = None
        self.weights_assignment = None
        self.X_active = None
        self.template_threshold = params["template_threshold"]

    @property
    def __name__(self):
        return "WDI_1NN"

    def fit(self, X, y):
        self.calc_templates(X, y)
        self.calc_weights()

    def predict_proba(self, X):
        # get vectors with active alarms per sample
        X_active = np.array(self.calc_active_alarms(X))
        y_proba = []
        # iterate over samples
        for i in range(X_active.shape[0]):
            y_assign_i = []
            # iterate over classes
            for c in self.classes:
                # calculate dissimilarity for assignment
                if sum(self.weights_assignment[c]) != 0.0:
                    y_assign_i.append(
                        sum(
                            self.weights_assignment[c]
                            * np.abs(X_active[i] - self.templates[c][1])
                        )
                        / sum(self.weights_assignment[c])
                    )
                else:
                    y_assign_i.append(0)
            # get class dissimilarities
            y_proba.append(y_assign_i)
        # convert dissimilarities to probabilities via softmax over negative distances
        y_proba = np.array(y_proba, dtype=float)
        logits = -y_proba
        logits = logits - np.max(logits, axis=1, keepdims=True)  # numerical stability
        exp_logits = np.exp(logits)
        y_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return np.array(y_proba)

    def calc_active_alarms(self, X):
        X_active = []
        for i in range(X.shape[0]):
            # transform in DataFrame
            df = pd.DataFrame(X[i].transpose())
            df_max = df.max()
            X_active.append([1 if df_max[j] == 1 else 0 for j in range(X.shape[1])])
        return np.array(X_active)

    def calc_templates(self, X, y):
        # get set of unique class labels in y
        self.classes = np.unique(y)
        # get vectors with active alarms per sample
        self.X_active = self.calc_active_alarms(X)
        # get the templates per class
        templates = {}
        for c in self.classes:
            sample_indices = np.where(y == c)
            class_frequency = sum(self.X_active[sample_indices]) / len(
                sample_indices[0]
            )
            class_template = np.where(class_frequency > self.template_threshold, 1, 0)
            templates[c] = [class_frequency, class_template]
        self.templates = templates

    def calc_weights(self):
        weights_assignment, weights_novelty, alpha_weights, beta_weights = (
            {},
            {},
            {},
            {},
        )
        # iterate over all templates / classes
        for c in self.classes:
            # calc alpha weights
            alpha_weights[c] = (self.templates[c][0] * self.templates[c][1]) + (
                (1 - self.templates[c][0]) * np.where(self.templates[c][1] == 1, 0, 1)
            )
        for c in self.classes:
            # calc beta weights
            beta_weights[c] = (sum(alpha_weights.values()) - alpha_weights[c]) / (
                self.classes.shape[0] - 1
            )
            # calc full weights
            weights_novelty[c] = 2 * alpha_weights[c] - 1
            weights_assignment[c] = (2 * alpha_weights[c] - 1) * (1 - beta_weights[c])

        self.weights_assignment = weights_assignment
        self.weights_novelty = weights_novelty
