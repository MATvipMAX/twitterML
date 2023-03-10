'''
    https://stackoverflow.com/questions/55272508/keras-how-to-write-customized-loss-function-to-aggregate-over-frame-level-predi
'''

import os
import time
import copy
import datetime
import numpy as np
import pandas as pd
from numpy.random import seed
from attention import AttentionWithContext
from utils import split_data
from utils.tokeniser import Tokenizer
from utils.utils import get_class_weights
from utils.embed import load_glove_tw_vectors
from utils.users import load_users, select_post_ana, combine_users
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize import TweetTokenizer
from keras.optimizers import Adam, AdaMod
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, concatenate
from keras.layers import Embedding, Dropout, Bidirectional, TimeDistributed
from keras.layers import CuDNNGRU, CuDNNLSTM, Conv1D
from keras.layers import BatchNormalization, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf
from tensorflow import set_random_seed


# time.sleep(3600)


def create_atte_model():
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                input_length=MAX_POST_LENGTH,
                                weights=[embedding_matrix],
                                trainable=False)

    sequence_input = Input(shape=(MAX_POST_LENGTH,))
    embedded_sequences = embedding_layer(sequence_input)
    l_lstm_sent = Bidirectional(CuDNNGRU(50, return_sequences=True))(embedded_sequences)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    l_lstm_sent = AttentionWithContext()(l_lstm_sent)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    preds = Dense(units=2, activation='softmax')(l_lstm_sent)
    sentEncoder = Model(sequence_input, preds)
    print(sentEncoder.summary())

    ana_input = Input(shape=(MAX_POSTS, len(i_data[0][0])))

    review_input = Input(shape=(MAX_POSTS, MAX_POST_LENGTH))
    l_lstm_sent = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = concatenate([l_lstm_sent, ana_input])
    l_lstm_sent = BatchNormalization()(l_lstm_sent)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    l_lstm_sent = Bidirectional(CuDNNGRU(16, return_sequences=True))(l_lstm_sent)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    l_lstm_sent = AttentionWithContext()(l_lstm_sent)
    l_lstm_sent = Dropout(0.2)(l_lstm_sent)
    preds = Dense(2, activation='softmax')(l_lstm_sent)
    model = Model([review_input, ana_input], preds)
    print(model.summary())

    adam = AdaMod()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])

    return model


SEED = 1
seed(SEED)
set_random_seed(SEED)

POST_SIZE = 100
MAX_POST_LENGTH = 18
MAX_POSTS = 500
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 50
NB_DEPRESS = 1984

DIR = '../data/'

inputs, labels = load_users(DIR, POST_SIZE, MAX_POSTS, NB_DEPRESS, True)
control = inputs[inputs.label == 0].reset_index(drop=True).copy()
depress = inputs[inputs.label == 1].reset_index(drop=True).copy()
del inputs, labels

liwc = pd.read_csv(DIR + 'LIWC2015-pron.csv', encoding='utf8')

embeddings_index = load_glove_tw_vectors(EMBEDDING_DIM)  # load pre-trained embedding vectors

i = 0
MODEL_FOLDER = "logs/{}/{}".format('ana', datetime.datetime.now().strftime("%Y-%m-%d %H-%M")+'p' + str(MAX_POSTS) + '-em' + str(EMBEDDING_DIM))
for data_index in split_data.split(depress, 4):
    data_fold = depress[depress.userid.isin(data_index)].copy()

    print(data_index[:10])
    print(data_fold.head())

    combine, labels = combine_users(control, data_fold)
    combine = combine.merge(liwc, on=['tweetid', 'userid'])
    inputs, i_inputs, labels = select_post_ana(combine, labels, MAX_POSTS)
    del combine, data_fold

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
    print('-----------{}-----------'.format(time.time() - start))

    print('Transforming text to input sequences')
    start = time.time()
    data = np.zeros((len(inputs), MAX_POSTS, MAX_POST_LENGTH), dtype='int32')
    i_data = np.zeros((len(inputs), MAX_POSTS, len(i_inputs[0][0])), dtype='float')

    for i, posts in enumerate(inputs):
        for j, post in enumerate(posts):
            if j < MAX_POSTS:
                sequences = tokenizer.texts_to_sequences([post])
                # sequences = tokenizer.lists_to_sequences([post])
                seq_data = pad_sequences(sequences, maxlen=MAX_POST_LENGTH)
                data[i, j] = seq_data
                i_data[i, :len(i_inputs[i])] = i_inputs[i]
    print('-----------{}-----------'.format(time.time() - start))

    del inputs, alltexts, i_inputs

    print('Finished loading data')
    print('Shape of data tensor:', data.shape)
    print('Shape of i_inputs tensor:', i_data.shape)
    print('Shape of label tensor:', labels.shape)

    print('Building embedding matrix')
    start = time.time()
    num_words = min(MAX_NB_WORDS, len(word_index) + 1)
    print('%s Selected word tokens' % num_words)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    # embedding_matrix = np.random.uniform(-1, 1, (num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print('-----------{}-----------'.format(time.time() - start))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
    for train_index, test_index in skf.split(np.asarray(data), labels[:, 1]):
        np.random.seed(0)
        tf.set_random_seed(0)
        sess = tf.Session(graph=tf.get_default_graph())
        K.set_session(sess)

        print(datetime.datetime.now())
        X_train, x_val = data[train_index], data[test_index]
        Xi_train, xi_val = i_data[train_index], i_data[test_index]
        y_train, y_val = labels[train_index], labels[test_index]
        del data, labels, i_data

        print('Number of positive and negative classes in training and validation set')
        print(y_train.sum(axis=0))
        print(y_val.sum(axis=0))

        class_weight = get_class_weights(np.asarray(y_train, 'int32')[:, 1])

        FOLDER = "{}/{}".format(MODEL_FOLDER, datetime.datetime.now().strftime("%Y-%m-%d %H-%M"))
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER)
        callbacks = TensorBoard(log_dir=FOLDER,
                                write_graph=True, write_grads=False, histogram_freq=0,
                                write_images=True, embeddings_freq=0, embeddings_layer_names='embedding_1',
                                embeddings_metadata=None)

        checkpoint = ModelCheckpoint(FOLDER + '/model.{epoch:02d}-{val_acc:.2f}.hdf5',
                                     verbose=0, monitor='val_acc',
                                     save_best_only=True, mode='auto')

        early_stopping = EarlyStopping(patience=3, monitor='val_acc', min_delta=0.001, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001,
                                      cooldown=0, min_lr=0.00000001)

        print("model fitting - Hierachical LSTM")

        print(class_weight)
        model = create_atte_model()

        model.fit([X_train, Xi_train], np.asarray(y_train, 'int32'), validation_data=([x_val, xi_val], np.asarray(y_val, 'int32')),
                  shuffle=False,
                  nb_epoch=200, batch_size=32, verbose=0, class_weight=class_weight,
                  callbacks=[checkpoint, callbacks])

        model.save(FOLDER + '/my_model.h5')
        del model, X_train, y_train, x_val, y_val, class_weight, Xi_train, xi_val
        K.clear_session()
        print(datetime.datetime.now())
        break
