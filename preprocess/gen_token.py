import pandas as pd
import numpy as np
import time
from utils.utils import clean_tweet

_dir = '../data/'
depress = pd.read_csv(_dir + 'final_replace_depress0220.csv', encoding='utf-8')
control = pd.read_csv(_dir + 'final_replace_control0220.csv', encoding='utf-8')

cdep = depress.clean.apply(clean_tweet)
ccon = control.clean.apply(clean_tweet)

depress['clean'] = cdep
control['clean'] = ccon

from nltk.tokenize import TweetTokenizer

tknzr = TweetTokenizer(reduce_len=True)
dalltexts = [tknzr.tokenize(text) for text in depress.clean.values]
calltexts = [tknzr.tokenize(text) for text in control.clean.values]

depress['clean'] = dalltexts
control['clean'] = calltexts

depress.to_csv(_dir + 'final_replace_depress0220token.csv', encoding='utf-8', index=False)
control.to_csv(_dir + 'final_replace_control0220token.csv', encoding='utf-8', index=False)
