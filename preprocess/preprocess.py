import pandas as pd
import re
from langdetect import detect_langs
from langdetect import DetectorFactory
from cleantext import clean

DetectorFactory.seed = 0
DIR = 'd:/New_datasets_062019/'

def check_ln(message):
    try:
        ln = detect_langs(message)
        for i, lang in enumerate(ln):
            if ln[i].lang == 'en' and ln[i].prob >= 0.8:
                result = 'en'
                return result
            else:
                result = 'non'
    except:
        result = 'non'
    return result


def remove_links(tweet):
    '''Takes a string and removes web links from it'''
    tweet = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*', 'url', tweet)  # remove http links
    #tweet = re.sub(r'bit.ly/\S+', 'url', tweet)  # rempve bitly links
    #tweet = tweet.strip('[link]')  # remove [links]
    return tweet


def remove_users(tweet):
    '''Takes a string and removes retweet and @user information'''
    tweet = re.sub('(RT\s@\w+)', 'rt user', tweet)  # remove retweet
    tweet = re.sub('(@\w+)', 'user', tweet)  # remove tweeted at
    return tweet


def hashtag(tweet):
    tweet = tweet.group()
    hashtag_body = tweet[1:]
    result = "<hashtag> {} ".format(hashtag_body.lower())
    return result

# Call first
# cleaning master function
def replace_tweet(tweet):
    # should remove &amp;
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = re.sub("#\S+", hashtag, tweet)
    tweet = tweet.lower()
    tweet = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "number", tweet)  # remove numbers
    tweet = re.sub("\n", " ", tweet)
    tweet = tweet.strip()
    tweet = clean(tweet, no_emoji=True)
    return tweet


# =====================================================================================================
# Pre-process Depresss
# =====================================================================================================

# d1 = pd.read_csv(DIR + 'depressed_tweets_0619.csv', encoding='utf-8')
# d2 = pd.read_csv(DIR + 'depressed_tweets_0419.csv', encoding='utf-8')
# d3 = pd.read_csv(DIR + 'depressed_tweets_0319.csv', encoding='utf-8')
# d4 = pd.read_csv(DIR + 'depressed_tweets_0219.csv', encoding='utf-8')
# d5 = pd.read_csv(DIR + 'depressed_tweets_0119.csv', encoding='utf-8')
# data = pd.concat([d1, d2, d3, d4, d5])
# del d1, d2, d3, d4, d5
def prepareTweets():
    data = pd.read_csv('preprocess/depressed_tweets.csv', encoding='utf-8')

    # nb_control = data.groupby(['userid'], as_index=False).count()[['userid', 'count']]
    # print(nb_control)
    # nb_control = nb_control[nb_control.tweet >= 100].copy()

    # data = data[data.userid.isin(nb_control.userid.values)].copy()

    # output = [check_ln(' '.join(data[data.userid == d].tweet)) for d in data.userid.unique()]

    # df = pd.DataFrame(data.userid.unique(), columns=['userid'])
    # df['ln'] = output

    # df = df[df.ln == 'en']
    # data = data[data.userid.isin(df.userid.values)]
    # data = data.copy()

    output = data.tweet.apply(replace_tweet)
    data['clean'] = output

    depress = data.copy()

    del data#, df, nb_control

    # print(depress)

    depress.sort_values(by=['userid'], inplace=True, ascending=False)
    depress[['tweet', 'userid', 'clean']].to_csv('C:/Users/mateu/OneDrive/Pulpit/data/final_replace_depress0220.csv', index=False, encoding='utf-8')

    # =====================================================================================================
    # Pre-process Control
    # =====================================================================================================

    # c1 = pd.read_csv(DIR + 'control_users_new062019_tweets_1.csv', encoding='utf8')
    # c2 = pd.read_csv(DIR + 'control_users_new062019_tweets_2.csv', encoding='utf8')
    # data = pd.concat([c1, c2])
    # del c1, c2
    data = pd.read_csv('preprocess/control_users.csv', encoding='utf8')


    # nb_control = data.groupby(['userid'], as_index=False).count()[['userid', 'tweet']]
    # nb_control = nb_control[nb_control.tweet >= 100].copy()

    # data = data[data.userid.isin(nb_control.userid.values)].copy()

    # output = [check_ln(' '.join(data[data.userid == d].tweet)) for d in data.userid.unique()]

    # df = pd.DataFrame(data.userid.unique(), columns=['userid'])
    # df['ln'] = output

    # df = df[df.ln == 'en']
    # data = data[data.userid.isin(df.userid.values)]
    # data = data.copy()

    output = data.tweet.apply(replace_tweet)
    data['clean'] = output

    control = data.copy()

    del data#, df, nb_control

    control.sort_values(by=['userid'], inplace=True, ascending=False)
    control[['tweet', 'userid', 'clean']].to_csv('C:/Users/mateu/OneDrive/Pulpit/data/final_replace_control0220.csv', index=False, encoding='utf-8')

    # =====================================================================================================
    # Pre-process Labelling
    # =====================================================================================================

    control_label = pd.DataFrame(control.userid.unique(), columns=['userid'])
    control_label['label'] = 'control'

    depress_label = pd.DataFrame(depress.userid.unique(), columns=['userid'])
    depress_label['label'] = 'depress'

    labels = pd.concat([control_label, depress_label]).reset_index(drop=True)

    labels.to_csv('final_replace_labels0220.csv', index=False, encoding='utf-8')
