
import numpy as np


def accuracy(y_real, y_predicted):
    """
    Returns the accuracy of given classification.
    """
    N = y_real.shape[0]
    correct = np.sum(y_predicted == y_real)
    return correct / N


def precision(y_real, y_predicted, c=1):
    """
    Returns the precision of given classification for class c.
    """
    tp = np.sum(np.logical_and(y_predicted == c, y_real == c))
    fp = np.sum(np.logical_and(y_predicted == c, y_real != c))
    return tp / (tp + fp)


def recall(y_real, y_predicted, c=1):
    """
    Returns the recall of given classification for class c.
    """
    tp = np.sum(np.logical_and(y_predicted == c, y_real == c))
    fn = np.sum(np.logical_and(y_predicted != c, y_real == c))
    return tp / (tp + fn)


def performance_binary(y_real, y_predicted, c=1):
    """
    Returns accuracy, precision and recall of given classification.
    """
    tp = np.sum(np.logical_and(y_predicted == c, y_real == c))
    fp = np.sum(np.logical_and(y_predicted == c, y_real != c))
    tn = np.sum(np.logical_and(y_predicted != c, y_real != c))
    fn = np.sum(np.logical_and(y_predicted != c, y_real == c))
    N = tp + fp + tn + fn
    accuracy = (tp + tn) / N
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return accuracy, precision, recall


def mean_average_precision(y_real, predicted_probs):
    """
    Returns the mean average precision of given classification.
    """
    y = y_real[predicted_probs.argsort()]
    N = y.shape[0]
    tp = np.sum(y)
    fp = N - tp
    s = 0
    for y_ in y:
        if y_ == 1:
            s += tp / (tp + fp)
        tp -= y_
        fp -= (1 - y_)
    return s / np.sum(y)
