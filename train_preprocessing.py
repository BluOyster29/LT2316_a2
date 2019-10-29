import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import argparse
from torch.utils.data import Dataset, DataLoader
from Data_set_Loader import DebatesSet

def read_csv(folder):
    '''
    Made this for testing as I have a folder with the csv files
    '''
    train_df = pd.read_csv(folder + 'split/train.csv')
    val_df = pd.read_csv(folder + 'split/validation.csv')
    test_df = pd.read_csv(folder + 'split/test.csv')
    return train_df, val_df, test_df

def fetchvocab(trainfile):
    df = trainfile

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

    int2char = dict(enumerate(vocab))
    char2int = {char : num for num, char in int2char.items()}
    return char2int

def encode_train_data(df, char2int):
    '''
    args:
        df - pandas dataframe
        char2int - dictionary with charager to integer representation
    simply goes through the dataframe and encodes to ingegral representations
    could be expanded to use pretrained vectors
    '''
    stop_words = set(stopwords.words('english'))
    for colname in ['sent1', 'sent2']:
        df[colname] = df[colname].apply(lambda x: [char2int[i] for i in x])

def zipping_data(df):
    '''
    args:
        df - pandas dataframe

    Goes through sentence1 and in dataframe and creates a tuple
    of each sentence so it can be passed to dataloader
    '''
    X_train = []
    y_train = []
    for idx, row in df.iterrows():
        sent_1 = row['sent1']
        sent_2 = row['sent2']
        X_train.append((sent_1,sent_2))
        y_train.append(row['label'])
    return X_train, y_train

if __name__== "__main__" :
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--trainfile', type=str, help='Name of csv file containing the training data.', required=True)
    #args = parser.parse_args()
    print("Getting Data sets")
    train_df, val_df, test_df = read_csv('data/')
    print("Generating vocab")
    vocab = fetchvocab(train_df)
    print("Getting training X and y")
    X_train, y_train = zipping_data(train_df)
    print("Encoding!")
    encoded = encode_train_data(train_df, vocab)
    print("Generating Dataloader")
    train_data = DebatesSet(X_train, y_train)
    train_loader = DataLoader(dataset=train_data,
                              batch_size=50,
                              shuffle=True)

    print(train_loader[0])
    print("Vocab Size: {}".format(len(vocab)))
    print("Number of Training Instances: {}".format(len(train_df)))
