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
3. `data_preprocessing.py` Reads the csv files 'data_splitting' pre/processes the data to be fed into a dataloader. Dataloader is then outputted to dataloader folder.
        - `--training_data`: path to training data 
	- `--testing_data`: path to testing data 
	- `--batch_size`: int to record batch size of data
	- `--output`: path to dataloader folder

**Training**

4. `train.py`: Currently trains rnn model and outputs the trained model to folder 
        - `-tr`: path to training dataloader
        - `-B`: batch size 
	- `-E`: Number of epochs for training 
	- `-M`: Model type, currently only rnn is working 
	- `-o`: Output path for trained model
	
**Testing**

5. `test.py`: Testing script, so far just for the rnn script. Prints to the screen the accuracy of the model

# To do (TBD)
1. Get vocab out of the training data - Done!
2. Convert sents to their integer representation and possibly GloVe or word2vec embeddings - Done - Can use word2vec too if we like
3. (up for discussion) Create a neat DataLoader-like object for fetching instances - Done! 
4. Training loop 
5. Feed into network!
6. ???
7. Profit
