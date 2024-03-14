import numpy as np

class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        for c in self.classes:
            X_c = X[y == c]
            self.parameters[c] = {
                'mean': X_c.mean(axis=0),
                'var': X_c.var(axis=0),
                'prior': X_c.shape[0] / X.shape[0]
            }
    
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        posteriors = []
        for c, params in self.parameters.items():
            prior = np.log(params['prior'])
            conditional = np.sum(np.log(self._pdf(x, params['mean'], params['var'])))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, x, mean, var):
        return (1. / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))
