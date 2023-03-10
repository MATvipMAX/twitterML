import regex as re
from collections import Counter
from gensim.models import Word2Vec


FLAGS = re.MULTILINE | re.DOTALL


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: float(majority/count) for cls, count in counter.items()}


def clean_str(txt):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    # FLAGS = re.MULTILINE | re.DOTALL
    txt = str(txt)
    txt = txt.strip().lower()
    txt = re.sub(r"\\", "", txt, flags=FLAGS)
    txt = re.sub(r"\'", "", txt, flags=FLAGS)
    txt = re.sub(r"\"", "", txt, flags=FLAGS)
    txt = re.sub(r"\r", " ", txt, flags=FLAGS)
    txt = re.sub(r"\n", " ", txt, flags=FLAGS)
    txt = re.sub(r"&gt;", " ", txt, flags=FLAGS)
    txt = re.sub(r"&lt;", " ", txt, flags=FLAGS)
    # txt = re.sub(r':)', "EMOSMILE", txt, flags=FLAGS)
    txt = re.sub(r"[0-9]+", "NUMBERS", txt, flags=FLAGS)
    txt = re.sub(r"[ ]+", " ", txt, flags=FLAGS)
    txt = re.sub(r'(.)\1{2,}', r'\1\1', txt, flags=FLAGS)
    txt = re.sub(r'http\S+', 'URL', txt, flags=FLAGS)
    return txt


def remove_nb(txt):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    # FLAGS = re.MULTILINE | re.DOTALL
    txt = str(txt)
    txt = re.sub(r"[0-9]+", "numbers", txt, flags=FLAGS)
    return txt


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} ".format(hashtag_body.lower())
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result


def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


def tweet_preprocess(text):
    # Different regex parts for smiley faces
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    # function so code less repetitive
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    text = re_sub(r"@\w+", "<user>")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text = re_sub(r"/", " / ")
    text = re_sub(r"<3", "<heart>")
    # text = re_sub(r"RT ", "<rt>")
    text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    text = re_sub(r"#\S+", hashtag)
    text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text = re_sub(r"\n", r" ")
    text = re_sub(r"\s+", r" ")
    # text = text.translate(str.maketrans('', '', string.punctuation))

    # -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
    # text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
    text = re_sub(r"([A-Z]){2,}", allcaps)

    return text.lower()


def remove_dot(text):
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    text = re_sub(r".+", r".")
    text = re_sub(r"?+", r"?")
    text = re_sub(r"&gt;", r">")
    text = re_sub(r"&lt;", r"<")

    return text


# cleaning master function
def clean_tweet(tweet):
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`–‘{|}~•@…“”’'
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    tweet = tweet.lower()  # lower case
    tweet = re.sub(r"&gt;", r" ", tweet)
    tweet = re.sub(r"&lt;", r" ", tweet)
    tweet = re.sub(r"&amp;", r" ", tweet)
    tweet = re.sub(r"\u200d", r" ", tweet)
    tweet = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " smile ", tweet)
    tweet = re.sub(r"{}{}p+".format(eyes, nose), " lolface ", tweet)
    tweet = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " sadface ", tweet)
    tweet = re.sub(r"{}{}[\/|l*]".format(eyes, nose), " neutralface ", tweet)
    tweet = re.sub(r"<3", " heart ", tweet)
    tweet = re.sub(r"[0-9]+", "numbers", tweet)
    tweet = re.sub('[' + my_punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = tweet.strip()
    return tweet


'''# cleaning master function
def clean_tweet(tweet):
    my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`–‘{|}\\\\~•@…“”’♂'
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    tweet = tweet.lower()  # lower case
    tweet = re.sub(r"&gt;", r" ", tweet)
    tweet = re.sub(r"&lt;", r" ", tweet)
    tweet = re.sub(r"&amp;", r" ", tweet)
    tweet = re.sub(r"\u200d", r" ", tweet)
    tweet = re.sub(r"[0-9]+", "numbers", tweet)
    tweet = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " smile ", tweet)
    tweet = re.sub(r"{}{}p+".format(eyes, nose), " lolface ", tweet)
    tweet = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " sadface ", tweet)
    tweet = re.sub(r"{}{}[\/|l*]".format(eyes, nose), " neutralface ", tweet)
    tweet = re.sub(r"<3", " heart ", tweet)
    tweet = re.sub('[' + my_punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = tweet.strip()
    return tweet
'''


'''my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@…“”’♂'

# cleaning master function
def clean_tweet(tweet):
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    tweet = tweet.lower()  # lower case
    tweet = re.sub(r"&gt;", r" ", tweet)
    tweet = re.sub(r"&lt;", r" ", tweet)
    tweet = re.sub(r"&amp;", r" ", tweet)
    tweet = re.sub(r"\u200d", r" ", tweet)
    tweet = re.sub(r"[0-9]+", "numbers", tweet)
    tweet = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "smile", tweet)
    tweet = re.sub(r"{}{}p+".format(eyes, nose), "lolface", tweet)
    tweet = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "sadface", tweet)
    tweet = re.sub(r"{}{}[\/|l*]".format(eyes, nose), "neutralface", tweet)
    tweet = re.sub(r"<3", "heart", tweet)
    tweet = re.sub('['+ my_punctuation + ']+', ' ', tweet)  # strip punctuation
    tweet = re.sub('\s+', ' ', tweet)  # remove double spacing
    tweet = tweet.strip()

    return tweet
'''


def load_w2v():
    return Word2Vec.load('../data/w2v.model')
