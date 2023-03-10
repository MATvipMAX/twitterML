import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve


def get_max_acc(y_test, probas_):
    fpr, tpr, thresholds = roc_curve(y_test, probas_)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_test,
                                             [1 if m > thresh else 0 for m in probas_]))
    accuracies = np.array(accuracy_scores)
    max_accuracy = accuracies.max()
    max_accuracy_threshold =  thresholds[accuracies.argmax()]
    return max_accuracy, max_accuracy_threshold


def find_optimal_cutoff(target, predicted):
    """
    Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])
