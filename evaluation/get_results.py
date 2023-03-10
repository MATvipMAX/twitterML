import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc


def get_max_acc(y_test, probas_):
    fpr, tpr, thresholds = roc_curve(y_test, probas_)
    accuracy_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_test,
                                              [1 if m > thresh else 0 for m in probas_]))
    accuracies = np.array(accuracy_scores)
    max_accuracy = accuracies.max()
    max_accuracy_threshold = thresholds[accuracies.argmax()]
    return max_accuracy, max_accuracy_threshold


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
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
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


results = []
for i in range(4):
    res1 = pd.read_csv('dl-35000-100' + str(i) + '.csv')

    y_test = res1.label.values.astype(int)
    probas_ = res1.class1.values
    max_accuracy, thresh = get_max_acc(y_test, probas_)
    print('=======================Max======================{:.2f}'.format(max_accuracy * 100))
    pred = [1 if m > thresh else 0 for m in probas_]
    print(classification_report(y_test, pred, target_names=['Control', 'Depress']))

    weighted = precision_recall_fscore_support(y_test, pred, average='weighted')
    macro = precision_recall_fscore_support(y_test, pred, average='macro')
    micro = precision_recall_fscore_support(y_test, pred, average='micro')
    both = precision_recall_fscore_support(y_test, pred)
    results.append(['dl-max-' + str(i), max_accuracy, weighted[0], weighted[1], weighted[2],
                    macro[0], macro[1], macro[2], micro[0], micro[1], micro[2],
                    both[0][0], both[1][0], both[2][0], both[0][1], both[1][1], both[2][1]])

    print('=======================Nor======================{:.2f}'.format(0.00 * 100))
    pred = [1 if m > 0.5 else 0 for m in probas_]
    max_accuracy = accuracy_score(y_test, pred)
    print(classification_report(y_test, pred, target_names=['Control', 'Depress']))

    weighted = precision_recall_fscore_support(y_test, pred, average='weighted')
    macro = precision_recall_fscore_support(y_test, pred, average='macro')
    micro = precision_recall_fscore_support(y_test, pred, average='micro')
    both = precision_recall_fscore_support(y_test, pred)
    results.append(['dl-nor-' + str(i), max_accuracy, weighted[0], weighted[1], weighted[2],
                    macro[0], macro[1], macro[2], micro[0], micro[1], micro[2],
                    both[0][0], both[1][0], both[2][0], both[0][1], both[1][1], both[2][1]])

    '''thresh = Find_Optimal_Cutoff(y_test, probas_)
    print('=========================Cutoff=========================')
    pred = [1 if m > thresh else 0 for m in probas_]
    print(classification_report(y_test, pred, target_names=['Control', 'Depress']))'''
#
#     weighted = precision_recall_fscore_support(y_test, pred, average='weighted')
#     macro = precision_recall_fscore_support(y_test, pred, average='macro')
#     micro = precision_recall_fscore_support(y_test, pred, average='micro')
#     both = precision_recall_fscore_support(y_test, pred)
#     results.append(['topic-lr-bl' + str(i), max_accuracy, weighted[0], weighted[1], weighted[2],
#                     macro[0], macro[1], macro[2], micro[0], micro[1], micro[2],
#                     both[0][0], both[1][0], both[2][0], both[0][1], both[1][1], both[2][1]])
#
results = pd.DataFrame(results, columns=['name', 'acc', 'pre-w', 'rec-w', 'f1-w',
                                         'pre-ma', 'rec-ma', 'f1-ma', 'pre-mi', 'rec-mi', 'f1-mi',
                                         'pre-c0', 'rec-c0', 'f1-c0', 'pre-c1', 'rec-c1', 'f1-c1']).sort_values(
    by=['name']).reset_index(drop=True)
print(results)
results.to_csv('dl-' + str(35000) + '-' + str(100) + '-results.csv', index=False)
