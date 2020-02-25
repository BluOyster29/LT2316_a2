import shutil, torch, pickle, os, argparse, torch
from tqdm import tqdm
from torch import nn
from torch.optim import Adam
import torch.optim as optim
from RNN_model import GRUNet
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(
        description="")
    parser.add_argument("-tr", "--train_dataloader", dest='train_dataloader', type=str, default="dataloaders/training",
                        help="Load dataloader")
    
    parser.add_argument("-B", "--batch_size", dest='batch_size', type=int, default=1,
                        help="Define the batch size for training")
    parser.add_argument("-E", "--nr_epochs", dest='nr_epochs', type=int,
                        default=2, help="Define the number of training epochs")
    parser.add_argument("-M", "--model_type", dest='model_type', type=str,
                        help='cnn or rnn')
    parser.add_argument("-o", "--model_output", dest='model_output', type=str,
                        help='cnn or rnn')

    args = parser.parse_args()
    return args

def load_dataloaders(filepath):


    with open('{}'.format(filepath), 'rb') as file:
        dataloaders = pickle.load(file)

    return dataloaders

def save_model(model, path):
    
    if os.path.exists(path) == False:
        os.mkdir(path)
        
    output_path = 'trained_model.pt'
    
    torch.save(model.state_dict(), '{}/{}'.format(path,output_path))

def get_vocab(path):
    with open(path, 'rb')as file:
        vocab = pickle.load(file)
    return vocab

def train(model, training_dl, vocab_size, device, nr_of_epochs, batch_size):

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    model.train()
    model = model.to(device)
    print('Training')
    epoch_nr = 0
    EPOCHS = list(range(nr_of_epochs))
    
    for i in tqdm(EPOCHS): 
        epoch_nr += 1
        epoch_loss = []
    
        for i in tqdm(training_dl):
            
            sent_1 = i[0][0].to(device)
            if sent_1.shape != torch.Tensor(batch_size,30).shape:
                continue
            sent_2 = i[0][1].to(device)
            optimizer.zero_grad()
            out = model(sent_1, sent_2)
            out = F.softmax(sum(out), dim=1)
            loss = criterion(out, i[1].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_loss = (sum(epoch_loss) / len(epoch_loss)) * 100
        print('Average Epoch Loss for Epoch {}: {}'.format(epoch_nr,avg_loss))

    return model 

def main(args):
    
    vocab_size = len(get_vocab('{}vocab.pkl'.format(args.train_dataloader[:-23])))
    device = 'cuda:01'
    training = load_dataloaders(args.train_dataloader)
    batch_size = args.batch_size
    input_dim = 30
    nr_of_epochs = args.nr_epochs
    input_dim = 30
    hidden_dim = 128
    output_dim = 2
    n_layers = 2
    print('Creating model')
    model = GRUNet(input_dim, hidden_dim, output_dim,
               n_layers, batch_size, vocab_size).to(device)
    print('Model Generated')
    print('Training Model with {} batches over {} epochs on {}'.format(batch_size,nr_of_epochs, device))
    trained_model = train(model, training, vocab_size, device, nr_of_epochs, batch_size)
    print('Outputting model to {}'.format(args.model_output))
    save_model(trained_model, args.model_output)
    return trained_model

if __name__ == '__main__':
    args = get_args()
    main(args)
