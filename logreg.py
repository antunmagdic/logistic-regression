
import numpy as np
import data


def _cross_entropy_loss(Y_oh, probs):
    N = Y_oh.shape[0]
    return -1/N * np.sum(np.log(np.array(
        [probs[i, np.argmax(Y_oh[i])] for i in range(N)])))


class LogisticRegression:
    """
    Logistic regression classifier.

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
    reg : {'l1', 'l2'}, defualt='l2'
        Determines whether L1 or L2 regularization will be used.
    eps : float, default=1E-6
        Training termination criterion. If a norm of a gradient during
        gradient descent is less than eps then training is finished.
    log : bool, default=False
        If set to True training progress will be printed to console.

    Attributes
    ----------
    W : ndarray of shape (n_features, n_classes)
        Matrix of weights used by this classifier. W[:, k] is a vector
        of weights associated with class k. n_features is the number of
        features and n_classes is the number of classes as given in fit
        method.
    b : ndarray of shape (n_classes,)
        Vector of biases used by this classifier. b[k] is a bias 
        associated with class k. n_classes is the number of classes as 
        given in fit method
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
        self._W, self._b = None, None

    @property
    def W(self):
        return self._W.copy()

    @property
    def b(self):
        return self._b.copy()

    def fit(self, X, y):
        """
        Fits this model to the given training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training datapoints. n_samples is the number of samples and
            n_features is the number of features.
        y : ndarray of shape (n_samples,)
            Target vector. Classes must be encoded by values 0,...,K-1
            where K is the number of classes.
        """
        Y_oh = data.as_one_hot(y)
        N, d, k = X.shape[0], X.shape[1], Y_oh.shape[1]
        W = np.random.randn(d, k)
        b = np.zeros(k)
        for i in range(self._max_iter):
            scores = X @ W + b
            e = np.exp(scores - np.max(scores, axis=1).reshape(N, 1))
            den = np.sum(e, axis=1)
            den.resize(N, 1)
            probs = e / den

            if self._log and i % 10 == 0:
                loss = _cross_entropy_loss(Y_oh, probs)
                if self._reg == self.L1:
                    loss += self._alpha * np.sum(np.abs(W))
                else:
                    loss += self._alpha * 1/2 * W.flatten() @ W.flatten()
                print('Iteration {}, loss = {}'.format(i, loss))

            Gs = probs - Y_oh
            grad_W = (1/N * Gs.T @ X).T
            grad_b = 1/N * np.sum(Gs, axis=0)

            if self._reg == self.L1:
                f = lambda wi: 1 if wi > 0 else -1
                grad_W += (self._alpha * np.vectorize(f)(W))
            else:
                grad_W += self._alpha * W

            W -= self._eta * grad_W
            b -= self._eta * grad_b

        self._W = W
        self._b = b

    def predict_proba(self, X):
        """
        Predicts the probabilities of given data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Array of datapoints. Classification probabilitiy of each 
            datapoint is computed and returned. n_samples is the number
            of samples and n_features is the number of features.

        Returns
        -------
        probs : ndarray of shape (n_samples, n_classes)
            Classification probabilities. probs[i, k] is the 
            probability of datapoint X[i] belonging to class encoded by
            k.
        """
        if self._W is None:
            raise RuntimeError(
                'fit must be called before a call to predict_proba')
        N = X.shape[0]
        scores = X @ self._W + self._b
        e = np.exp(scores - np.max(scores, axis=1).reshape(N, 1))
        den = np.sum(e, axis=1)
        den.resize(N, 1)
        return e / den

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
            datapoint X[i].
        """
        if self._W is None:
            raise RuntimeError('fit must be called before a call to predict')
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def _print_performance(y, y_predicted):
    print('Accuracy: {:.5f}'.format(accuracy(y, y_predicted)))


def _get_range(X, margin=0.2):
    xmin, ymin = tuple(np.min(X, axis=0))
    xmax, ymax = tuple(np.max(X, axis=0))
    xrange = (xmin - margin * (xmax-xmin), xmax + margin * (xmax-xmin))
    yrange = (ymin - margin * (ymax-ymin), ymax + margin * (ymax-ymin))
    return xrange, yrange


def _plot_results(clf, X, y, y_predicted):
    xrange, yrange = _get_range(X)
    plot_surface(lambda x: clf.predict(x), xrange=xrange, yrange=yrange, 
                 offset=0.5, n=1000, levels=(0, 1, 2, 3, 4))
    plot_classification(X, y, y_predicted)
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])
    plt.show()


def main():
    seed = np.random.randint(1E9)
    print('Using seed {}.'.format(seed))
    np.random.seed(seed)

    X, y = sample_gaussian_2d(3, 100)
    clf = LogisticRegression(alpha=0.5, eta=0.05, max_iter=1000)
    clf.fit(X, y)
    y_predicted = clf.predict(X)

    _print_performance(y, y_predicted)
    _plot_results(clf, X, y, y_predicted)


if __name__ == '__main__':
    from data import sample_gaussian_2d
    from metrics import accuracy
    from plotting import plot_classification, plot_surface
    import matplotlib.pyplot as plt
    main()
    