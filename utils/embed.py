import numpy as np


# GLOVE_DIR = "D:/Research/"
GLOVE_DIR = "C:/Users/mateu/OneDrive/Pulpit/data/"


def load_glove_tw_vectors(dims):
    print('Loading embedding vectors from a file')
    embeddings_index = {}
    f = open(GLOVE_DIR + 'glove.twitter.27B.' + str(dims) + 'd.txt', 'r', encoding='utf-8')
    for line in f:
        try:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except ValueError:
            print("error on line" + line)
    f.close()

    print('Total %s word vectors.' % len(embeddings_index))

    return embeddings_index
