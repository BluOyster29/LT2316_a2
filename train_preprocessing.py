import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import argparse
from torch.utils.data import Dataset, DataLoader

def fetchvocab(trainfile):
    df = pd.read_csv(trainfile)

    stop_words = set(stopwords.words('english'))
    tokenize = lambda x: word_tokenize(x)

    print('Removing stopwords')
    for colname in ['sent1', 'sent2']:
        df[colname] = df[colname].str.lower().str.split() #t his could be tokenized in a better way of course :)
        df[colname] = df[colname].apply(lambda x: [item for item in x if item not in stop_words])

    print('Getting sentences and vocab')
    vocab = set()
    sents = []
    for idx, row in df.iterrows():
        sent1 = row['sent1']
        vocab.update(sent1)
        sent2 = row['sent2']
        vocab.update(sent2)
        label = row['label']
        sents.append([sent1, sent2, label])

    class SpeechLoader():
        def __init__(self, trainfile):
            # Convert sentences to idx


if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainfile', type=str, help='Name of csv file containing the training data.', required=True)
    args = parser.parse_args()

    trainfile = args.trainfile
    fetchvocab(trainfile)