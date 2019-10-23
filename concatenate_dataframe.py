import pandas as pd
from glob import glob
import os
import argparse

def concatenate_csv(directory, file_out):
    #all_debates = glob(directory + '/*.csv')[:50] #let's not use aaaall the files just yet
    all_debates = glob(directory + '/*.csv') #let's not use aaaall the files just yet
    df = pd.concat((pd.read_csv(f, index_col=None,names=['sent1', 'sent2', 'class']) for f in all_debates))

    print('Writing full dataframe to {}.csv'.format(file_out))
    df.to_csv(file_out + '.csv', index=None)


if __name__== "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_folder', type=str, help='Name of a folder containing csv files.', required=True)
    parser.add_argument('--full_dataframe', type=str, help='Name of file to write dataframe to.', required=True)

    args = parser.parse_args()

    concatenate_csv(args.csv_folder, args.full_dataframe)
