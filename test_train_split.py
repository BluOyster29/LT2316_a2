import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def fetchwords(df, num_sentences, column1, column2):
    '''Fetch and label all sentences for training as well as the unique vocabulary
    sents is a list of 3tuples containing ([sent1], [sent2], label)
    '''

    df = pd.read_csv(df)

    def split_data(df, trainfile, valfile):
        '''train, val: strings
        '''
        train, val = train_test_split(df, test_size=0.3)
        train = train.groupby('class').apply(lambda x: x.sample(n=int(num_sentences/2))).reset_index(drop=True)
        print('Writing out train file to {}.csv.'.format(trainfile))
        train.to_csv(trainfile+'.csv', index=False)
        print('Writing out train file to {}.csv.'.format(valfile))
        val.to_csv(valfile+'.csv', index=False)


    split_data(df, 'train', 'val')

if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_file', type=str, help='Name of the file containing the data frame', required=True)
    parser.add_argument('--num_sentences', type=int, help='The number of sentences to fetch', default=20000)
    #parser.add_argument('--full_df', type=str, help='Name of file to write df to.', required=True)

    args = parser.parse_args()

    fetchwords(args.df_file, args.num_sentences, 'sent1', 'sent2')