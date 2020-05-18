
import numpy as np


class BinaryLogisticRegression:
    """
    Binary logistic regression classifier.

    Uses gradient descent to determine the parameters of the logistic
    regression classifier.

    Parameters
    ----------
    max_iter : int, default=1000
        Maximum number of iterations allowed in gradient descent.
    eta : float, default=0.1
        Learning rate of gradient descent. Must be greater than 0.
    alpha : float, default=0.1
        Regularization hyperparameter. Larger alpha results in stronger
        regularization. Should be greater than 0.
    reg : {'l1', 'l2'}, default='l2'
        Determines whether L1 or L2 regularization will be used.
    eps : float, default=1E-6
        Training termination criterion. If a norm of a gradient during
        gradient descent is less than eps then training is finished.
    log : bool, default=False
        If set to True training progress will be printed to console.

    Attributes
    ----------
    w : ndarray of shape (n_features,)
        Vector of weights in the decision function. n_features is the 
        number of features as given in fit method.
    b : float
        Bias used in the decision function.
    """

    L1 = 1
    L2 = 2

    def __init__(self, max_iter=1000, eta=0.1, alpha=0.1, reg='l2', eps=1E-6, 
                 log=False):
        self._max_iter = max_iter
        self._eta = eta
        self._alpha = alpha
        self._reg = None
        if reg == 'l2':
            self._reg = self.L2
        elif reg == 'l1':
            self._reg = self.L1
        else:
            raise ValueError('invalid reg value: ' + reg)
        self._eps = eps
        self._log = log
        self._w, self._b = None, None

    @property
    def w(self):
        return self._w.copy()

    @property
    def b(self):
        return self._b

    def fit(self, X, y):
        """
        Fits this model to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training datapoints. n_samples is the number of samples and
            n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target vector. Since this is a binary classifier classes 
            must be encoded by values 0 and 1.
        """
        N, d = X.shape
        w = np.random.randn(d)
        b = 0
        for i in range(self._max_iter):
            scores = X @ w + b
            e = np.exp(scores)
            probs = e / (1 + e)

            if self._log and i % 10 == 0:
                loss = -1/N * np.sum(np.log(np.abs(1 - y - probs)))
                if self._reg == self.L1:
                    loss += self._alpha * np.sum(np.abs(w))
                else:
                    loss += self._alpha * 1/2 * w @ w
                print('Iteration {}, loss = {}'.format(i, loss))

            gs = probs - y
            # .T not needed since gs is a 1D array
            grad_w = 1/N * gs.T @ X
            grad_b = 1/N * np.sum(gs)
            
            if self._reg == self.L1:
                grad_w += (self._alpha * 
                    np.array([1 if wi > 0 else -1 for wi in w]))
            else:
                grad_w += self._alpha * w

            if np.linalg.norm(np.append(grad_w, grad_b)) < self._eps:
                break

            w -= self._eta * grad_w
            b -= self._eta * grad_b
        
        self._w = w
        self._b = b

    def decision_function(self, X):
        """
        Returns the values of the decision function for given data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Datapoints for which the decision function values should be
            computed. n_samples is the number of samples and n_features
            is the number of features.
        
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Decision function values of given datapoints.
        """
        return X @ self._w + self._b

    def predict_proba(self, X):
        """
        Predicts the probabilities of given data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Array of datapoints. Classification probabilitiy of each 
            datapoint is computed and returned. n_samples is the 
            number of samples and n_features is the number of features.

        Returns
        -------
        probs : ndarray of shape (n_samples, 2)
            Classification probabilities. probs[i, 0] is the 
            probability of the datapoint X[i] belonging to class 
            encoded by 0 and probs[i, 1] is the probability of 
            datapoint X[i] belonging to class encoded by 1.
        """
        if self._w is None:
            raise RuntimeError(
                'fit must be called before a call to predict_proba')
        scores = self.decision_function(X)
        scores.resize(X.shape[0], 1)
        e = np.exp(scores)
        probs = e / (1 + e)
        return np.hstack((1 - probs, probs))

    def predict(self, X):
        """
        Predicts the classes of given datapoints.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Datapoints to be classified. n_samples is the number of 
            samples and n_features is the number of features.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Vector of predicted classes. y[i] is the class of the 
            datapoint X[i]. Classes are encoded by 0s and 1s (as given
            in fit method).
        """
        if self._w is None:
            raise RuntimeError('fit must be called before a call to predict')
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(np.int)


def _print_performance(y, y_predicted):
    accuracy, precision, recall = performance_binary(y, y_predicted)
    print('Accuracy:  {:.5f}'.format(accuracy))
    print('Precision: {:.5f}'.format(precision))
    print('Recall:    {:.5f}'.format(recall))


def _get_range(X, margin=0.2):
    xmin, ymin = tuple(np.min(X, axis=0))
    xmax, ymax = tuple(np.max(X, axis=0))
    xrange = (xmin - margin * (xmax-xmin), xmax + margin * (xmax-xmin))
    yrange = (ymin - margin * (ymax-ymin), ymax + margin * (ymax-ymin))
    return xrange, yrange


def _plot_results(clf, X, y, y_predicted):
    xrange, yrange = _get_range(X)
    plot_surface(lambda x: clf.predict_proba(x)[:, 1],
                 xrange=xrange, yrange=yrange, offset=0.5)
    cmap = lambda x: (0.2, 0.2, 0.2, 1) if x == 0 else (1, 1, 1, 1)
    plot_classification(X, y, y_predicted, cmap=cmap)
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])
    plt.show()


def main():
    seed = np.random.randint(1E9)
    print('Using seed {}.'.format(seed))
    np.random.seed(seed)

    X, y = sample_gaussian_2d(2, 100)
    clf = BinaryLogisticRegression(alpha=0.2)
    clf.fit(X, y)
    y_predicted = clf.predict(X)

    w, b = clf.w, clf.b
    print('w = ({:.3f}, {:.3f}), b = {:.3f}'.format(w[0], w[1], b))
    _print_performance(y, y_predicted)
    _plot_results(clf, X, y, y_predicted)


if __name__ == '__main__':
    from data import sample_gaussian_2d
    from metrics import performance_binary
    from plotting import plot_classification, plot_surface
    import matplotlib.pyplot as plt
    main()
