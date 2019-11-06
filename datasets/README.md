## Pytorch Dataset and DataLoader

### Dataset
The dataset can be iterated through and each index consists of a tuple and an integer. The tuple is a vectorised sequence corrosponding to sentence 1 and sentence 2 and the integer is either 0 or 1. 1 being the label of a change of speaker.

### DataLoader
The dataloader is an iterable dataset consisting of batches from the dataset. I believe the batch size is 1, this is so you can see excactly what is going on in there. If you run the data_preprocessing script and change the batch size before running you can make it larger, I will add an arg at some point. The batch is also shuffled. This confuses me slightly because each sentence in the dataset may not have the same label. Need to work this out
