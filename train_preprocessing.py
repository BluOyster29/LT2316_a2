from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import pandas as pd
import argparse

def preprocessing(df, num_sentences):
    df = pd.read_csv(df)


    print('Splitting dataset into equal amounts of change/same label.')
    df = df.groupby('class').apply(lambda x: x.sample(n=int(num_sentences/2))).reset_index(drop=True)

    # tokenize and remove stopwords from all sentences
    stop_words = set(stopwords.words('english'))

    for colname in ['sent1', 'sent2']:
        df[colname] = df[colname].str.lower().str.split()
        df[colname] = df[colname].apply(lambda x: [item for item in x if item not in stop_words])

    # # print('removing stopwords')
    # df[
    # df['sent1'].apply(lambda x: [item for item in x if item not in stop_words]) &
    # df['sent2'].apply(lambda x: [item for item in x if item not in stop_words]) 
    # ] 

    # print('preparing vocab')
    # vocab = set()
    # df['sent1'].str.lower().str.split().apply(vocab.update)
    # df['sent2'].str.lower().str.split().apply(vocab.update)
    # vocab = list(vocab)

    # this is not too pretty
    print('Getting sentences and vocab')
    vocab = set()
    sents = []
    for idx, row in df.iterrows():
        sent1 = row['sent1']
        vocab.update(sent1)
        sent2 = row['sent2']
        vocab.update(sent2)
        label = row['class']
        sents.append([sent1, sent2, label])

    # print('preparing vocab')
    # for sent1, sent2, label in sents:
    #     vocab.update(sent1)
    #     vocab.update(sent2)

if __name__== "__main__" :
    parser = argparse.ArgumentParser(description="This script makes a 50/50 split between classes and removes stopwords from the data.")
    parser.add_argument('--train_data', type=str, help='Name of the csv containing the training data', default='train.csv')
    parser.add_argument('--num_sentences', type=int, help='The number of sentences to fetch', default=20000)
    args = parser.parse_args()
    preprocessing(args.train_data, args.num_sentences)