from collections import OrderedDict
from collections import defaultdict
import multiprocessing as mp
import twikenizer
from nltk.tokenize import TweetTokenizer


# def token_text(arg, **kwarg):
#     return Tokenizer.token_text(*arg, **kwarg)
def token_text(twk, text):
    # twk = twikenizer.Twikenizer()
    res = twk.tokenize(text)
    return res


class Tokenizer(object):
    def __init__(self, num_words=None,
                 lower=True,
                 oov_token=None,
                 document_count=0,
                 **kwargs):
        self.num_words = num_words
        self.word_counts = OrderedDict()
        self.word_docs = defaultdict(int)
        self.index_docs = defaultdict(int)
        self.document_count = document_count
        # self.twk = twikenizer.Twikenizer()
        self.twk = TweetTokenizer(reduce_len=True)
        self.oov_token = oov_token
        self.word_index = dict()
        self.index_word = dict()
        self.lower = lower

    def token_text(self, text):
        return self.twk.tokenize(text)

    def fit_on_texts(self, texts):
        # pool = mp.Pool(processes=8)
        # results = [pool.apply_async(token_text, args=(text.lower(),)) for text in texts]
        # output = [p.get() for p in results]
        # pool.close()
        # pool.join()
        # texts = output

        # output = []
        # for text in texts:
        #     p = mp.Process(target=self.token_text, args=(text,))
        #     output.append(p)
        # texts = [x.start() for x in output]

        # for text in output:
        for text in texts:
            seq = text
            # seq = self.twk.tokenize(text.lower())
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

        self.index_word = dict((c, w) for w, c in self.word_index.items())

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def texts_to_sequences(self, texts):
        seq = []
        for text in texts:
            # seq = self.twk.tokenize(text.lower())
            seq.append(self.twk.tokenize(text.lower()))
        # return list(self.texts_to_sequences_generator(texts))
        return list(self.texts_to_sequences_generator(seq))

    def lists_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
        #     seq = self.twk.tokenize(text.lower())
            seq = text


            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect
