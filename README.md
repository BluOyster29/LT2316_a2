# LT2316_a2

Group Cabbage!

**Preprocessing:**

1.  `generate_data.py`: Reads XML files with debates and saves the splitted sentences to a CSV file with columns *sent1, sent2, label, speaker_id*. The label says if there is a speaker change between the two sentences *(label 1)* or not *(label 0)*.
	- `--debate_path`: *Path to folder which holds the debate XML files.*
	- `--data_path`: *Name of folder where the new file should be saved.*
2. `data_splitting.py`: Splits data into train/validation/test files. *split* defines the size of the training set. The rest is equally divided into a validation and a test set.
	- `--df_file_path`: *Path to the file containing the data frame.*
	- `--split`: *Define the train split. The test split will be 1 minus this value.*
	- `--data_file_path`: *Path to folder to which the new train/validation/test files should be written.*
3. 'data_preprocessing' Reads the csv files 'data_splitting' pre/processes the data to be fed into a dataloader. No args at the moment but maybe plan on adding some. After this program then we are ready to start a training loop. 

**Training**

# Current pipeline – feel free to change this!

* Maybe Torchtext in the future
* Fix Data splitting, at the moment the sizes are not ideal 
* Training
* Evaluating 

# To do (TBD)
1. Get vocab out of the training data - Done!
2. Convert sents to their integer representation and possibly GloVe or word2vec embeddings - Done - Can use word2vec too if we like 
3. (up for discussion) Create a neat DataLoader-like object for fetching instances - Done! 
4. Training loop 
5. Feed into network!
6. ???
7. Profit
