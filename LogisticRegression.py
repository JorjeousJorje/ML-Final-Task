import numpy as np
from numba import njit, prange
    
    
    
class LogisticRegression:
    def __init__(self, eta=.01, n_iter=10000, tol=0.0001, random_state=0, weight_decay=0, reg_type="l2"):
        self._eta = eta
        self._n_iter = n_iter
        self._tol = tol
        self.__random_state = random_state
        self._reg_type = reg_type
        self._weight_decay = weight_decay
        
    @staticmethod
    def _sigmoid(X, w):
        denominator = 1.0 + np.exp(-np.dot(X, w))
        return 1.0 / denominator
    
    def __reg_grad(self):
        if self._reg_type == "l2":
            return 2 * self._weight_decay * self.w_
        elif self._reg_type == "l1":
            return self._weight_decay * np.sign(self.w_)
        else:
            raise TypeError('Unknown reg_type: {self._reg_type}.')
        
    def predict(self, X_test, proba=False, threshold=.5):
        if not self.__fit:
            raise TypeError('The model has not been fit.')
        if self.__multiclass:
            # Extract the probabilities from each of the fitted models.
            probas = map(lambda model: model.predict(X_test, proba=True)[:, [1]], self.models_)
            # Tie them together.
            self.proba_ = np.concatenate(list(probas), axis=1)
            # Make the prediction: a class with the highest
            # probability is chosen as the predicted class.
            self.y_hat_ = np.array([self._classes[idx] for idx in self._classes[np.argmax(self.proba_, axis=1)]])
            if proba:
                return np.concatenate([self.y_hat_[:, np.newaxis], self.proba_], axis=1)
            else:
                return self.y_hat_
        else:
            self._X_test = np.array(X_test, dtype=np.dtype('float64'))
            # Normalize the testing data.
            if self.__normalize:
                self._X_test = (self._X_test - self.__Xmin) / self.__Xrange
            # Append the bias term.
            self._X_test = np.concatenate([np.ones(shape=(self._X_test.shape[0], 1)), self._X_test], axis=1)
            # Calculate the probabilities.
            self.proba_ = self._sigmoid(self._X_test, self.w_)
            self.y_hat_ = np.zeros(shape=(self.proba_.shape[0], ))
            self.y_hat_[self.proba_ >= threshold] = 1
            if proba:
                return np.concatenate([self.y_hat_[:, np.newaxis], self.proba_[:, np.newaxis]], axis=1)
            else:
                return self.y_hat_
            
    def __handle_multiclass_fit(self):
        self.__multiclass = True
        self.models_ = []
        for class_ in self._classes:
            y = np.zeros(shape=self._y.shape)
            y[self._y == class_] = 1
            # Initialize and fit the model.
            lr = LogisticRegression(
                eta=self._eta,
                n_iter=self._n_iter,
                tol=self._tol,
                reg_type = self._reg_type,
                random_state=self.__random_state,
                weight_decay=self._weight_decay
            )
            # One vs All fit
            lr.fit(self._X, y)

            if self.__normalize:
                lr.__normalize = self.__normalize
                lr.__Xmin = self.__Xmin
                lr.__Xrange = self.__Xrange
            self.models_.append(lr)
        self.__fit = True
        return self
    
    def __handle_binary_fit(self, X):
        self.__multiclass = False
        self._X = np.concatenate(
            [np.ones(shape=(len(X), 1)), self._X],
            axis=1
        )
        # Generate the initial weights.
        rs = np.random.RandomState(seed=self.__random_state)
        self.w_ = rs.normal(size=(self._X.shape[1], ))
        # Gradient descent
        for _ in range(self._n_iter):
            grad = np.dot(self._sigmoid(self._X, self.w_) - self._y, self._X) + self.__reg_grad()
            self.w_ -= self._eta * grad
            if all(np.absolute(grad) < self._tol):
                break
        self.intercept_ = self.w_[0]
        self.coef_ = self.w_[1:]
        self.__fit = True
        return self
    
    def fit(self, X, y, normalize=True):
        self._X = np.array(X, dtype=np.dtype('float64'))
        self._y = np.array(y, dtype=np.dtype('int64')).squeeze()

        self.__normalize = normalize
        if self.__normalize:
            self.__Xmin = self._X.min(axis=0)
            self.__Xrange = self._X.max(axis=0) - self.__Xmin
            self._X = (self._X - self.__Xmin) / self.__Xrange
        # Check if the problem is multiclass:
        self._classes = np.unique(self._y)
        if len(self._classes) > 2:
            return self.__handle_multiclass_fit()
        else:
            return self.__handle_binary_fit(X)