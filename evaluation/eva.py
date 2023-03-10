import os
import time
import datetime
import numpy as np
import pandas as pd
from utils.users import load_users, select_post, combine_users
from utils.metrics import find_optimal_cutoff, get_max_acc
from attention import AttentionWithContext
from keras.optimizers import AdaMod
from utils.tokeniser import Tokenizer, token_text
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from numpy.random import seed
from tensorflow import set_random_seed
from nltk.tokenize import TweetTokenizer
from utils import split_data
from keras.models import load_model
import glob
import tensorflow as tf
from keras import backend as K
import sys


SEED = 1
seed(SEED)
set_random_seed(SEED)

POST_SIZE = 100
MAX_POST_LENGTH = 18
MAX_POSTS = 500
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100
NB_DEPRESS = 1984

DIR = '../data/'

inputs, labels = load_users(DIR, POST_SIZE, MAX_POSTS, NB_DEPRESS, True)
control = inputs[inputs.label == 0].reset_index(drop=True)
depress = inputs[inputs.label == 1].reset_index(drop=True)

fold = 0
results = []
MODEL_FOLDER = "2020-02-16 20-59ran-500-em100"
MODEL_FOLDER = "../mil/logs/LSTM/{}/".format(MODEL_FOLDER)
MODEL_FOLDER = sorted(glob.iglob(os.path.join(MODEL_FOLDER, '*')), key=os.path.getctime, reverse=False)

for data_index in split_data.split(depress, 4):
    data_fold = depress[depress.userid.isin(data_index)].copy()

    print(data_index[:10])
    print(data_fold.head())

    combine, labels = combine_users(control, data_fold)
    inputs, labels = select_post(combine, labels, MAX_POSTS)

    # transform Y to categories
    print(labels.label.value_counts())
    labels = labels.label.values
    labels = to_categorical(np.asarray(labels))

    print('Input number: ', len(inputs))
    print('Label number: ', len(labels))

    # alltexts = np.hstack(np.array(inputs))
    alltexts = np.hstack(np.array(inputs).flatten())

    print('Tokenizing text')
    start = time.time()
    tknzr = TweetTokenizer(reduce_len=True)
    alltexts = [tknzr.tokenize(text.lower()) for text in alltexts]

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(alltexts)
    word_index = tokenizer.word_index
    print('Total %s unique tokens.' % len(word_index))
    elapsed_time_fl = (time.time() - start)
    print(elapsed_time_fl)

    print('Transforming text to input sequences')
    start = time.time()
    data = np.zeros((len(inputs), MAX_POSTS, MAX_POST_LENGTH), dtype='int32')

    for i, posts in enumerate(inputs):
        for j, post in enumerate(posts):
            if j < MAX_POSTS:
                # sequences = tokenizer.lists_to_sequences([post])
                sequences = tokenizer.texts_to_sequences([post])
                seq_data = pad_sequences(sequences, maxlen=MAX_POST_LENGTH)
                data[i, j] = seq_data
    elapsed_time_fl = (time.time() - start)
    print(elapsed_time_fl)

    del inputs, alltexts

    print('Finished loading data')

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
    for train_index, test_index in skf.split(np.asarray(data), labels[:, 1]):
        np.random.seed(0)
        tf.set_random_seed(0)
        sess = tf.Session(graph=tf.get_default_graph())
        K.set_session(sess)

        print(datetime.datetime.now())
        X_train, x_val = data[train_index], data[test_index]
        y_train, y_val = labels[train_index], labels[test_index]

        model_name = sorted(glob.iglob(os.path.join(MODEL_FOLDER[fold], '*.hdf5')), key=os.path.getctime, reverse=True)[0]
        # model_name = sorted(glob.iglob(os.path.join(MODEL_FOLDER[fold], '*')), key=os.path.getctime, reverse=True)[1]
        print(model_name)
        model = load_model(model_name,
                           custom_objects={'AttentionWithContext': AttentionWithContext, 'optimizer': AdaMod})
        probas_ = model.predict(x_val)
        y_test = y_val[:, 1].astype(int)
        max_accuracy, thresh = get_max_acc(y_test, probas_[:, 1])
        pred = [1 if m > thresh else 0 for m in probas_[:, 1]]
        print('Fold {} accuracy = {:.2f}'.format(fold+1, max_accuracy*100))
        print(classification_report(y_test, pred, target_names=['Control', 'Depress']))

        df = pd.DataFrame(y_test, columns=['label'])
        df['class0'] = probas_[:, 0]
        df['class1'] = probas_[:, 1]
        df.to_csv('mil-' + str(MAX_POSTS) + '-' + str(EMBEDDING_DIM) + str(fold) + '.csv', index=False)

        weighted = precision_recall_fscore_support(y_test, pred, average='weighted')
        macro = precision_recall_fscore_support(y_test, pred, average='macro')
        micro = precision_recall_fscore_support(y_test, pred, average='micro')
        both = precision_recall_fscore_support(y_test, pred)
        results.append(['mil-' + str(MAX_POSTS) + '-' + str(EMBEDDING_DIM) + '-' + str(fold),
                        max_accuracy, weighted[0], weighted[1], weighted[2],
                        macro[0], macro[1], macro[2], micro[0], micro[1], micro[2],
                        both[0][0], both[1][0], both[2][0], both[0][1], both[1][1], both[2][1]])

        fold += 1

        break

results = pd.DataFrame(results, columns=['name', 'acc', 'pre-w', 'rec-w', 'f1-w',
'pre-ma', 'rec-ma', 'f1-ma', 'pre-mi', 'rec-mi', 'f1-mi',
'pre-c0', 'rec-c0', 'f1-c0', 'pre-c1', 'rec-c1', 'f1-c1']).sort_values(by=['name']).reset_index(drop=True)
print(results)
results.to_csv('mil-' + str(MAX_POSTS) + '-' + str(EMBEDDING_DIM) + '-results.csv', index=False)
