import numpy as np


def split_train_test(data, labels, fold):
    c_data, c_labels = len(data), len(labels)
    assert c_data == c_labels
    nb = int(c_data/fold)
    train = []
    test = []
    for i in range(1,fold+1):
        start, end = nb*i-nb, nb*i
        train.append(list(range(start, int(start+(nb*0.8)))))
        test.append(list(range(int(end-(nb*0.2)), end)))
    return train, test


def split(data, fold):
    data = data.userid.unique()
    c_data = len(data)
    np.random.seed(0)
    data = np.random.choice(data, size=c_data, replace=False)
    nb = int(c_data / fold)
    data_index = []
    for i in range(1, fold + 1):
        start, end = nb * i - nb, nb * i
        data_index.append(data[start: end])
    return data_index
