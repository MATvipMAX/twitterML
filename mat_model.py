from preprocess.preprocess import prepareTweets
from mil.lstm_model_full import getModel

if __name__ == '__main__':
    prepareTweets()
    getModel()