# LT2316_a2
Group Cabbage!

**Preprocessing:**

1.  `data_preprocessing.py`: Reads XML files with debates and saves the splitted sentences to a CSV file with columns *sent1, sent2, label, speaker_id*. The label says if there is a speaker change between the two sentences *(label 1)* or not *(label 0)*.
   - `--debate_path`: *Path to folder which holds the debate XML files.*
   - `--data_path`: *Name of folder where the new file should be saved.*
2. `data_splitting.py`: Splits data into train/validation/test files. *split* defines the size of the training set. The rest is equally divided into a validation and a test set.
   - `--df_file_path`: *Path to the file containing the data frame.*
   - `--split`: *Define the train split. The test split will be 1 minus this value.*
   - `--data_file_path`: *Path to folder to which the new train/validation/test files should be written.*





# Current pipeline – feel free to change this!

* Use torchtext for the above step instead!

# To do (TBD)
1. Get vocab out of the training data
2. Convert sents to their integer representation and possibly GloVe or word2vec embeddings
3. (up for discussion) Create a neat DataLoader-like object for fetching instances
4. Feed into network!
5. ???
6. Profit
