import pandas as pd, argparse, pickle, re, string, numpy as np, torch, random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Data_set_Loader import DebatesSets
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--training_data', dest='train', type=str, help='Path to training data csv format', default='data/train.csv')
    parser.add_argument('--testing_data', type=str, dest='test',help='Path for testing data', default='data/testing.csv')
    parser.add_argument('--batch_size', dest='batch_size', type=int, help='Batch size for training', default=100)
    args = parser.parse_args()
        
    return args

def tokenize(data_frame):

    '''
    Loops that go through the dataframe and tokenizes the strings
    Using regex to split hyphenated words and nltk word_tokenize for Tokenizing
    Input is a data frame
    '''

    sent1_tokenized = []
    labels = []
    print("Fetching Labels")
    for i in data_frame['class']:
        labels.append(i)

    print("Tokenizing sentence 1")
    for i in tqdm(data_frame['sent_1']):
        sent1_tokenized.append([re.sub('-',' ',x.lower()) for x in word_tokenize(i)])

    sent2_tokenized = []
    print("Tokenizing sentence 2")
    for i in tqdm(data_frame['sent_2']):
        sent2_tokenized.append([re.sub('-',' ',x.lower()) for x in word_tokenize(i)])

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

    print('Post Processing sentence 1')
    for i in tqdm(sent1_tokenized):
        i = remove_stopwords(i)

    print('Post Processing sentence 2')

    for i in tqdm(sent2_tokenized):
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
    
    # return the max len (=len of longest sequence), to be used for padding
    max1 = df.sent1.str.len().max()
    max2 = df.sent2.str.len().max()
    max_len = max([max1, max2])
    
    return df, max_len

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

    return new_vocab_emb, int2char

def encode(sent1,vocab):
    '''
    Code that encodes words to vocab integers
    '''
    encoded_matrix = []
    for i in sent1:
        encoded_vector = []
        for x in i[:30]:
            try:
                encoded_vector.append(vocab[x])
            except:
                encoded_vector.append(random.randint(0, len(vocab)))
                                     
        encoded_matrix.append(torch.LongTensor(encoded_vector))

    return pad_sequence(encoded_matrix, batch_first=True, padding_value=0)

def process(dataframe, train, vocab):
    
    sent1_tokenized, sent2_tokenized, labels = tokenize(dataframe)
    #proc_sent1, proc_sent2 = post_processing(sent1_tokenized, sent2_tokenized)
    if train == True:
        vocab = gen_vocab(sent1_tokenized, sent2_tokenized)
        
    encoded_sent1 = encode(sent1_tokenized,vocab)
    encoded_sent2 = encode(sent2_tokenized, vocab)
    dataset = DebatesSets(encoded_sent1, encoded_sent2, labels)
    
    if train == True:
        return dataset, vocab
    return dataset

def main(args):
    print('Loading Dataframes')
    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    print('Train, Test loaded')
    train_dataset, vocab = process(train_df, train=True, vocab=None)
    test_dataset = process(test_df, train=False, vocab=vocab)

    #glove_vocab_dict = pickle.load(open('pretrained_embeddings/glove/glove_vocab.pkl', 'rb'))
    #embeddings = generate_glove_vocab(glove_vocab_dict, vocab)
   
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    with open('dataloaders{}pkl'.format(args.train[4:-3]), 'wb') as output:
        print('Pickling Training dataloader to: {}'.format(output))
        pickle.dump(train_loader, output)
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    with open('dataloaders{}pkl'.format(args.test[4:-3]), 'wb') as output:
        print('Pickling Testing Dataloader to: {}'.format(output))
        pickle.dump(test_loader, output)
        
if __name__ == '__main__':
    args = get_args()
    main(args)
    



