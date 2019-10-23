from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def fetchwords(dataframe, column1, column2):
    '''Fetch and label all sentences for training as well as the unique vocabulary
    sents is a list of 3tuples containing ([sent1], [sent2], label)
    '''

    dataframe = pd.read_csv(dataframe)

    # stop_words = set(stopwords.words('english'))
    #ocab = list(filter(lambda x: x not in stop_words, vocab))

    # print('removing stopwords')
    # dataframe[
    # dataframe['sent1'].apply(lambda x: x not in stop_words) &
    # dataframe['sent2'].apply(lambda x: x not in stop_words) 
    # ] 

    # sents = []
    # for idx, row in dataframe.iterrows():
    #     sents.append([word_tokenize(row[column1]), word_tokenize(row[column2]), row['class']])

    #vocab = set()
    #dataframe[column1].str.lower().str.split().apply(vocab.update)
    #vocab = list(vocab)
    #print('vocab length:', len(vocab))

    #sentences, uniquevocab = fetchwords(df, 'sent1', 'sent2')

    #print('no. sentences:', len(sents))


    # def remove_stopwords(sent):
    #     return list(filter(lambda x: x not in stop_words, sent))

    # i = 0
    # for a, b, label in sents:
    #     sents[i][0] = remove_stopwords(a)
    #     sents[i][1] = remove_stopwords(b)
    #     i += 1

    # vocab = remove_stopwords(vocab)
    print('size:', dataframe.size)

    #print(dataframe.head())
    def split_data(dataframe, trainfile, valfile):
        '''train, val: strings
        '''
        train, val = train_test_split(dataframe, test_size=0.3)
        print('Writing out train file to {}.csv.'.format(trainfile))
        train.to_csv(trainfile+'.csv', index=False)
        print('Writing out train file to {}.csv.'.format(valfile))
        val.to_csv(valfile+'.csv', index=False)


    split_data(dataframe, 'train', 'val')
    #print('size:', dataframe.size)
    #print(vocab[:100])
    #print(sents[:10])

    # idx = {num: word for num,word in enumerate(vocab)}
    # idx = {word:num for num,word in idx.items()}
    # print(idx)

    # dataframe[
    # dataframe['sent1'].apply(lambda x: [idx[y] for y in x]) &
    # dataframe['sent2'].apply(lambda x: [idx[y] for y in x]) 
    # ] 

if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_file', type=str, help='Name of the file containing the data frame', required=True)
    #parser.add_argument('--full_dataframe', type=str, help='Name of file to write dataframe to.', required=True)

    args = parser.parse_args()

    fetchwords(args.df_file, 'sent1', 'sent2')