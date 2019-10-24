# LT2316_a2
Group Cabbage!

# Current pipeline – feel free to change this!

data_preprocessing.py: reads XML files and saves them to a csv with columns (sent1, sent2, speakerid?, class)
concatenate_dataframe: concatenates debate csv:s into one MEGA dataframe
test_train_split: you may guess what this does! (it splits MEGA into training and testing csvs.)

# To do (TBD)
1. Get vocab out of the training data
2. Convert sents to their integer representation and possibly GloVe or word2vec embeddings
3. (up for discussion) Create a neat DataLoader-like object for fetching instances
4. Feed into network!
5. ???
6. Profit
