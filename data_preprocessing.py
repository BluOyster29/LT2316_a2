import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle, re, string
from Data_set_Loader import DebatesSets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def read_csv(folder):
    # reads the pre-split csvs and returns pandas dataframes
    train_df = pd.read_csv(folder + 'split/train.csv')
    val_df = pd.read_csv(folder + 'split/validation.csv')
    test_df = pd.read_csv(folder + 'split/test.csv')
    return train_df, val_df, test_df

def tokenize(data_frame):

    '''
    Loops that go through the dataframe and tokenizes the strings
    Using regex to split hyphenated words and nltk word_tokenize for Tokenizing
    Input is a data frame
    '''

    counter = 0
    sent1_tokenized = []
    tenp = len(data_frame) / 5
    ten = 0
    labels = []
    print("Fetching Labels")
    for i in data_frame['label']:
        labels.append(i)

    print("Tokenizing sentence 1")
    for i in data_frame['sent1']:
        if counter % tenp == 0:
            print(str(ten) + '% tokenized')
            ten += 10
        sent1_tokenized.append([re.sub('-',' ',x.lower()) for x in word_tokenize(i)])
        counter +=1

    sent2_tokenized = []
    counter = 0
    print("Tokenizing sentence 2")
    for i in data_frame['sent2']:
        if counter % tenp == 0:
            print(str(ten) + '% tokenized')
            ten += 10
        sent2_tokenized.append([re.sub('-',' ',x.lower()) for x in word_tokenize(i)])
        counter +=1

    print("Tokens Generated")

    return sent1_tokenized, sent2_tokenized, labels

def remove_stopwords(tokenized_sent):
    '''
    from a tokenised sentence the loop removes any left over punctuation
    or stop words and some numbers. can be expanded with further tokens
    to remove.
    '''
    stop_words = stopwords.words('english')
    stop_words += string.punctuation
    for i in tokenized_sent:
        if i in stop_words or type(i) == int:
            tokenized_sent.remove(i)

    return tokenized_sent

def post_processing(sent1_tokenized, sent2_tokenized):
    '''
    Master post processing script for remobing stop words
    '''

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

def gen_vocab(sent1_tokenz, sent2_tokenz):

    '''
    Generates a vocab, word to index in vocab
    '''

    print('Generating Vocab')
    raw_text = []
    for i in sent1_tokenz:
        raw_text += i
    for i in sent2_tokenz:
        raw_text += i

    int2char = dict(enumerate(set(raw_text)))
    vocab = {char : num for num, char in int2char.items()}
    return vocab

def create_df(sent1,sent2,labels):

    '''
    script for outputting csv which is the post processed dataframe
    '''

    df = pd.DataFrame(data=list(zip(sent1,sent2,labels)), columns=['sent1','sent2','labels'])
    df.to_csv('data/processed_training.csv')
    return df

def generate_glove_vocab(glove_vocab, vocab):

    '''
    creates a vocab dictionary using glove embeddings
    input: dictionary containing glove embeddings
           index vocab
    output: vocab matching words to glove embeddings
    code adapted from:
        https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    '''
    print('Generating Glove Embeddings')
    new_vocab = {}
    weights_matrix=np.zeros((len(vocab), 50))
    words_found = 0
    words_not_found = 0
    for i, word in enumerate(vocab):
        try:
            new_vocab[word] = glove_vocab[word]
            weights_matrix[i] = glove_vocab[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))
            words_not_found +=1
            new_vocab[word] = weights_matrix[i]

    print('Words found: {}'.format(words_found))
    print('Glove embeddings make up {}% of vocabulary'.format(round(words_found / words_not_found * 100)))
    int2char = {num : char for char, num in vocab.items()}
    new_vocab_emb = {vocab[char] : num for char, num in new_vocab.items()}

    return new_vocab_emb

def encode(sent1,vocab):
    '''
    Code that encodes words to vocab integers
    '''
    encoded_matrix = []
    for i in sent1:
        encoded_vector = []
        encoded_vector.append(torch.LongTensor([vocab[x] for x in i[:30]]))
        encoded_matrix += encoded_vector

    return pad_sequence(encoded_matrix, batch_first=True, padding_value=0)

if __name__ == '__main__':
    print('Loading Dataframes')
    train_df, val_df, test_df = read_csv('data/')
    print('Train, Val, Test loaded')
    train_sent1_tokenized, train_sent2_tokenized, labels = tokenize(train_df)
    proc_sent1, proc_sent2 = post_processing(train_sent1_tokenized, train_sent2_tokenized)
    vocab = gen_vocab(proc_sent1, proc_sent2)
    glove_vocab_dict = pickle.load(open('pretrained_embeddings/glove/glove_vocab.pkl', 'rb'))
    embeddings = generate_glove_vocab(glove_vocab_dict, vocab)
    encoded_sent1 = encode(proc_sent1,vocab)
    encoded_sent2 = encode(proc_sent2, vocab)
    print('Encoding data with embeddings')
    master_set = DebatesSets(encoded_sent1, encoded_sent2, labels)
    print('Outputting training dataset and dataloader')
    with open('datasets/train_dataset.pkl', 'wb') as output:
        pickle.dump(master_set, output)
    master_loader = DataLoader(master_set, batch_size = 1, shuffle=True)
    with open('datasets/train_dataloadert.pkl', 'wb') as output:
        pickle.dump(master_loader, output)


    '''To do:
                add args for batch_size
                add tokenizing args
                args for file location
    '''
