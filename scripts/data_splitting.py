import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

def split_data(df, data_path , train_split):
    '''
    Splits data into train, validation and test sets. All files are balanced with equal amount of each label.
    :param df: dataframe holding all instances
    :param data_path: path to which the new train/test files should be written
    :param train_split: percentage of train split
    '''

    # create new folder for the instances files
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # read dataframe
    df = pd.read_csv(df)

    # test split
    test_split = (1-train_split)

    # group dataframe by classes for balanced splits
    group_df = df.groupby('label')
    # number of change instances so we can make balanced sets
    num_change = int(group_df['label'].value_counts()[1])
    classes = [group_df.get_group(0).sample(num_change), group_df.get_group(1)]

    # holds all dataframes per label
    train_sets = []
    test_sets = []
    val_sets = []

    # iterate over labels

    for label in tqdm(classes):
        # create train/test split
        train, test = train_test_split(label, test_size=test_split)
        train_sets.append(train)
        # split test set into two equal sized sets for validation and test
        val, test = train_test_split(test, test_size=0.5)
        test_sets.append(test)
        val_sets.append(val)

    # concatenate the two dataframes per label and shuffle the data
    train = pd.concat(train_sets).sample(frac=1).reset_index(drop=True)
    test = pd.concat(test_sets).sample(frac=1).reset_index(drop=True)
    val = pd.concat(val_sets).sample(frac=1).reset_index(drop=True)

    # save train set
    print('Writing out train file to {}.csv.'.format(data_path))
    train.to_csv(os.path.join(data_path, 'train.csv'), index=False)

    print('train: ', len(train))
    print(train['label'].value_counts())

    # save test and validation set
    print('Writing out validation and test file to {}.csv.'.format(data_path))
    val.to_csv(os.path.join(data_path, 'validation.csv'), index=False)
    test.to_csv(os.path.join(data_path, 'test.csv'), index=False)

    print('test: ', len(test))
    print(test['label'].value_counts())
    print('val: ', len(val))
    print(val['label'].value_counts())

if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_file_path', type=str, help='Path to the file containing the data frame.', default='data/debates_sents.csv')
    #parser.add_argument('--num_sentences', type=int, help='The number of sentences to fetch.', default=20000)
    parser.add_argument('--split', type=float, help='Define the train split.', default=0.6)
    parser.add_argument('--data_file_path', type=str, help='Path of folder to which the new train/validation/test files should be written.', default='data')

    args = parser.parse_args()

    # get arguments
    df_file_path = args.df_file_path
    #num_sentences = args.num_sentences
    train_split = args.split
    data_file_path = args.data_file_path

    # split data
    split_data(df_file_path, data_file_path, train_split)