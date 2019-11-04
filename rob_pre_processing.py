import pandas as pd
from pandas import read_csv
import string

def read_csv(folder):
    #needs args
    train_df = pd.read_csv(folder + 'split/train.csv')
    val_df = pd.read_csv(folder + 'split/validation.csv')
    test_df = pd.read_csv(folder + 'split/test.csv')
    return train_df, val_df, test_df 

def tokenize(data_frame):
    counter = 0
    sent1_tokenized = []
    tenp = len(data_frame) / 5
    ten = 0

    print("Tokenizing sentence 1")
    for i in data_frame['sent1']:
        if counter % tenp == 0:
            print(str(ten) + '% tokenized')
            ten += 10
        sent1_tokenized.append(word_tokenize(i))
        counter +=1 

    sent2_tokenized = []
    counter = 0
    print("Tokenizing sentence 2")
    for i in data_frame['sent2']:
        if counter % tenp == 0:
            print(str(ten) + '% tokenized')
            ten += 10
        sent2_tokenized.append(word_tokenize(i))
        counter +=1 

    print("Tokens Generated")
    
    return sent1_tokenized, sent2_tokenized

def remove_stopwords(tokenized_sent):
    stopwords = stopwords.words('english')
    for i in tokenized_sent:
        if i in stopwords or type(i) == int:
            tokenized_sent.remove(i)

    return tokenized_sent
    
def post_processing(sent1_tokenized, sent2_tokenized):
    counter = 0
    ten = 0
    tenper = len(sent1_tokenized) / 5

    print('Post Processing sentence 1')
    for i in sent1_tokenized:
        counter += 1
        if counter % tenper == 0:
            print(str(ten) + '% processed')
            ten += 10
        i = remove_stopwords(i)

    print('Post Processing sentence 2')
    counter = 0
    for i in sent2_tokenized:
        counter += 1
        if counter % tenper == 0:
            print(str(ten) + '% processed')
            ten += 10
        i = remove_stopwords(i)
    
    print('Finished')
    
    return sent1_tokenized, sent2_tokenized

train_df, val_df, test_df = read_csv('data/')

train_sent1_tokenized, train_sent2_tokenized = tokenize(train_df)
