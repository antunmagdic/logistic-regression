
import numpy as np


class RandomGaussian2D:
    """
    Generates random samples from randomized 2D Gaussian distribution.

    Parameters
    ----------
    xmin : float, default=0
        Minimum x value of mean to be randomized.
    xmax : float, default=10
        Maximum x value of mean to be randomized.
    ymin : flaot, default=0
        Minimum y value of mean to be randomized.
    ymax : float, default=10
        Maximum y value of mean to be randomized.
    """

    def __init__(self, xmin=0, xmax=10, ymin=0, ymax=10):
        ranges = np.array([xmax - xmin, ymax - ymin])
        offsets = np.array([xmin, ymin])
        self._mean = np.random.random_sample(2) * ranges + offsets

        # covariance matrix eigen values
        D = np.diag(np.random.random_sample(2))
        # rotation angle
        phi = np.random.random_sample() * 2*np.pi
        R = np.array([
            [np.cos(phi), -np.sin(phi)], 
            [np.sin(phi),  np.cos(phi)]
        ])
        self._cov = R.T @ D @ R

    def get_sample(self, N):
        """
        Generates a random sample from this distribution.

        Parameters
        ----------
        N : int
            Number of samples to generate.

        Returns
        -------
        X : ndarray of shape (N, 2)
            Matrix with N samples. Every row in returned matrix is one 
            generated random sample.
        """
        return np.random.multivariate_normal(self._mean, self._cov, N)

        
def sample_gaussian_2d(C, N, xmin=0, xmax=10, ymin=0, ymax=10):
    """
    Generates random Gaussian samples from multiple classes.

    Parameters
    ----------
    c : int
        Number of classes. For each class, samples are drawn from a 
        random 2D Gaussian distribution created for the corresponding
        class.
    N : int
        Number of samples to generate from each class.
    xmin : float, default=0
        Minimum x value of Gaussian means to be randomized.
    xmax : float, default=10
        Maximum x value of Gaussian means to be randomized.
    ymin : float, default=0
        Minimum y value of Gaussian means to be randomized.
    ymax : float, default=10
        Maximum x value of Gaussian means to be randomized.

    Returns
    -------
    (X, y) : (ndarray of shape (C * N, 2), ndarray of shape (C * N,))
        X is an array of datapoints. Each row represents a datapoint 
        and y[i] is the class of datapoint X[i].
    """
    X = np.empty((C * N, 2))
    y = np.empty(C * N, dtype=np.int)
    for i in range(C):
        g = RandomGaussian2D()
        X_ = g.get_sample(N)
        y_ = i * np.ones(N)
        X[i*N:(i+1)*N, :] = X_
        y[i*N:(i+1)*N] = y_
    return X, y
    

def as_one_hot(y):
    """
    Returns a one-hot encoding of given vector of classes.

    Parameters
    ----------
    y : ndarray of shape (N,)
        A vector of classes. Classes should be encoded by value from 
        0 to C-1 (both included), where C is the number of classes. N 
        is the number of samples in y.

    Returns
    -------
    y_one_hot : ndarray of shape (N, C)
        One-hot encoding of given vector. y[i] represents a one-hot 
        encoding of y[i]. N is the number of samples and C is the 
        number of classes.
    """
    k = np.max(y) + 1
    Y_oh = np.zeros((y.shape[0], k), dtype=np.int)
    for i, c in enumerate(y):
        Y_oh[i, c] = 1
    return Y_oh
    