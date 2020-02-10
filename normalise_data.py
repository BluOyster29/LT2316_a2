import pandas as pd, argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--training_data', dest='train', type=str, help='Path to training data csv format', default='data/train.csv')
    
    parser.add_argument('--testing_data', type=str, dest='test',help='Path for testing data', default='data/testing.csv')
    
    parser.add_argument('--training_output', dest='train_out', type=str, help='file for training output', default='data/unnamed!.csv')
    
    parser.add_argument('--testing_output', dest='test_out', type=str, help='file for testing output', default='data/unnamed!.csv')
    
    parser.add_argument('--output_data', dest='output', type=bool, help='output data to csv', default=False)
    
    
    args = parser.parse_args()
    
    return args

def find_max_examples(data):
    
    classes = {'true' : 0,
              'false' : 0}
    
    for i in data['class']:
        if i == 0:
            classes['true'] += 1
        elif i == 1:
            classes['false'] += 1
    
    return min(classes.values())

def normalise_data(max_examples,data):
    
    examples = []
    counter = 0
    for i in zip(data['sent_1'], data['sent_2'],data['class']):
        if counter == max_examples:
            
            break
        elif i[2] == 1:
            counter += 1
            examples.append(i)
        else:
            continue
            
    counter = 0
    for i in zip(data['sent_1'], data['sent_2'],data['class']):
        if counter == max_examples:
            break
        elif i[2] == 0:
            counter += 1
            examples.append(i)
        else:
            continue
        
    data = {'sent_1' : [],
            'sent_2' : [],
            'class'  : []}
    
    for i in examples:
        data['sent_1'].append(i[0])
        data['sent_2'].append(i[1])
        data['class'].append(i[2])
        
    return pd.DataFrame(data=data)
    
def count_examples(dataframe):
    true = 0
    false = 0
    
    for i in dataframe['class']:
        if i == 0:
            false += 1
        elif i == 1:
            true += 1
        
    print('Total true classes: {}'.format(true))
    print('Total false classes: {}'.format(false))
    
def output(train_data, train_path, test_data, test_path):
    print('Outputting training data to: {}'.format(train_path))
    train_data.to_csv(train_path)
    print('Outputting testing data to: {}'.format(test_path))
    test_data.to_csv(test_path)
    
if __name__ == '__main__':
    
    args = get_args()
    print(args.output)
    print('Opening Training dataframe')
    train_df = pd.read_csv(args.train)
    print('Opening Testing dataframe')                   
    test_df = pd.read_csv(args.test)
    training_maxnum = find_max_examples(train_df)
    testing_maxnum = find_max_examples(test_df)
    print('Generating normalised training data')                   
    normalised_train_df = normalise_data(training_maxnum, train_df)
    print('Generating normalised testing data')                    
    normalised_test_df = normalise_data(testing_maxnum, test_df)
    count_examples(normalised_train_df)
    count_examples(normalised_test_df)
    if args.output == True:
        output(normalised_train_df, args.train_out, normalised_test_df, args.test_out)
    