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
    parser.add_argument("-M", "--trained_model", dest='trained_model', type=str,
                        help="select trained model")
    parser.add_argument("-T", "--test_loader", dest='test_loader', type=str,
                        help="select trained model")
    parser.add_argument("-B", "--batch_size", dest='batch_size', type=int,
                        help="batch size of trained model")
    
    args = parser.parse_args()
    return args

def test_model(model, testing_dl, args):
    device = 'cuda:01'
    count = 0
    correct = 0
    print('Testing Model')
    model.eval().to(device)
    for i in tqdm(testing_dl):

        sent_1 = i[0][0].to(device)
        
        if sent_1.shape != torch.Tensor(args.batch_size,30).shape:
            continue
            
        sent_2 = i[0][1].to(device)
        y = i[1].to(device)

        prediction = sum(model(sent_1, sent_2))

        for x, actual in zip(prediction,y):
            
            count += 1
            _, indeces = torch.max(x.data, dim=0)
            if indeces.item() == actual:
                correct += 1
        

    accuracy = correct/count * 100
    
    return accuracy 

def load_dataloaders(filepath):

    with open('{}'.format(filepath), 'rb') as file:
        dataloaders = pickle.load(file)

    return dataloaders

def load_model(path, vocab):
    
    with open(path, 'rb') as input_model:
        data = torch.load(input_model)
    trained_model = GRUNet(input_dim=30, hidden_dim=128, output_dim=2,
               n_layers=2, batch_size=args.batch_size, vocab_size=len(vocab))
    
    trained_model.load_state_dict(data)
    return trained_model

def get_vocab(path):
    with open(path, 'rb')as file:
        vocab = pickle.load(file)
    return vocab

def main(args):
    testing_dl = load_dataloaders(args.test_loader)
    vocab = get_vocab('vocab/vocab.pkl')
    trained_model = load_model(args.trained_model, vocab)
    accuracy = test_model(trained_model, testing_dl, args)
    print('Accuracy for model: {}'.format(accuracy))
    
if __name__ == '__main__':
    args = get_args()
    main(args)
    
    
                                        
                                        
                                        
        
                

    