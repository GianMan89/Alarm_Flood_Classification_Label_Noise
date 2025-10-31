class AFC:
    def __init__(
        self,
        classifier,
        clf_params={},
        random_state=42,
        step_list=[10],
        stepwise=False,
    ):
        self.classifier = classifier
        self.clf_params = clf_params
        self.random_state = random_state
        self.step_list = step_list
        self.stepwise = stepwise
        self.clf_dict = {}

    @property
    def __name__(self):
        return "AFC"

    def fit(self, X, y):
        # If stepwise is enabled, iterate over X using step_size as step and fit a classifier for each step
        if self.stepwise:
            for i in self.step_list:
                X_step = X[:, :, :i]
                self.clf_dict[i] = self.classifier(self.clf_params)
                self.clf_dict[i].fit(X_step, y)
        else:
            self.clf_dict[0] = self.classifier(self.clf_params)
            self.clf_dict[0].fit(X, y)

    def predict_proba(self, X, step=None):
        if self.stepwise == False:
            y_scores = self.clf_dict[0].predict_proba(X)
        else:
            y_scores = self.clf_dict[step].predict_proba(X)
        return y_scores
