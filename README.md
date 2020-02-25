# LT2316_a2

Group Cabbage!

All python scripts are in the folder scripts, run the scripts in the root directory. 

**Preprocessing:**

1.  `generate_data.py`: Reads XML files with debates and saves the splitted sentences to a CSV file with columns *sent1, sent2, label, speaker_id*. The label says if there is a speaker change between the two sentences *(label 1)* or not *(label 0)*. The name of the folder is where all the output of the scripts will belong.
	- `--debate_path`: *Path to folder which holds the debate XML files.*
	- `--data_path`: *Name of folder where the new file should be saved.*
	
	e.g `python scripts/generate_data.py --debate_path /usr/local/courses/lt2316-h19/a2/hansard/scrapedxml/debates/ --data_path experiment_1`
	
2. `data_splitting.py`: Splits data into train/validation/test files. *split* defines the size of the training set. The rest is equally divided into a validation and a test set.
	- `--df_file_path`: *Path to the csv file.*
	- `--split`: *Define the train split. The test split will be 1 minus this value.*
	- `--data_file_path`: *Path to folder to which the new train/validation/test files should be written.*
	
	e.g `python scripts/data_splitting.py --df_file_path experiment_1/debates_sents.csv --split 0.75 --data_file_path experiment_1/split_data`

3. `data_preprocessing.py` Reads the csv files 'data_splitting' pre/processes the data to be fed into a dataloader. Dataloader is then outputted to dataloader folder.
        - `--training_data`: path to training data 
	- `--testing_data`: path to testing data 
	- `--batch_size`: int to record batch size of data
	- `--output`: path to dataloader folder
	
	e.g `python scripts/data_preprocessing.py --training_data experiment_1/split_data/train.csv --testing_data experiment_1/split_data/test.csv --batch_size 200 --output experiment_1/dataloaders`

**Training**

4. `train.py`: Currently trains rnn model and outputs the trained model to folder 
        - `-tr`: path to training dataloader
        - `-B`: batch size 
	- `-E`: Number of epochs for training 
	- `-M`: Model type, currently only rnn is working 
	- `-o`: Output directory for trained model
	
	e.g `python scripts/train.py -tr experiment_1/dataloaders/training_dataloader.pkl -B 200 -E 1 -o experiment_1/trained_models`

**Testing**

5. `test.py`: Testing script, so far just for the rnn script. Prints to the screen the accuracy of the model
	- `-M`: Path to the trained model
        - `-B`: batch size of dataset 
	- `-T`: Path to the test set dataloader 
	
	e.g `python scripts/test.py -M experiment_1/trained_models/trained_model.pt -T experiment_1/dataloaders/testing_dataloader.pkl -B 200`


# To do (TBD)
1. Get the CNN working 
2. Write up the experiment 
3. ???
4. Profit!


