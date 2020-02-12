import pickle
from torch import nn
import torch
from numpy import random
import csv
from torch import optim
from CNN_model import CNNClassifier
import numpy as np
import argparse

#dev = torch.device("cuda:{}".format(hash('gusstrlip') % 4) if torch.cuda.is_available() else "cpu")
data = pickle.load(open('datasets/train_dataset.pkl', 'rb'))
train_loader = pickle.load(open('datasets/train_dataloadert.pkl', 'rb'))
dev = torch.device("cuda")

def trainCNN(model, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()
    model = model.to(dev)
    model.set_dev(dev)
    for epoch in range(1, num_epochs+1):
        losses = 0
        for i, data in enumerate(train_loader, 0):
            #if i == len(train_loader)-1:
            #    break
            inputs, labels = data
            #print(inputs[0].shape)
            if (inputs[0].shape)[0] != 30:
                break
            labels = labels.to(dev)
            optimizer.zero_grad()
            pred = model(inputs)
            loss = criterion(pred, labels)
            #print('instance no.:', i, 'loss:', loss.item())
            losses += loss.item()
            loss.backward()
            optimizer.step()
            if i == len(train_loader)-2:
                print('Average loss at epoch {}: {}'.format(epoch, losses/i-1))

    return model

def getweights(vocab, int2char):
    embeddings_dict = {}
    with open("glove.6B.50d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector
            else:
                pass
    print('Fetching embedding vectors')

    embeddings = {}
    for item in int2char:
        word = int2char[item]
        try:
            vec = embeddings_dict[word]
            embeddings.update({item: vec})
        except:
            embeddings.update({item: random.rand(50).astype(np.float32)})

    weights = list(embeddings.values())
    weights = torch.tensor(weights, dtype=torch.float, requires_grad=False)
    print(type(weights))
    return weights

def train_model():
    #if pretrained == True:
    W = getweights(data.vocab, data.int2char)
    model = CNNClassifier(batch_size=30, initial_num_channels=50, num_classes=2, num_channels=30, weights=W)
    #model = CNNClassifier(batch_size=30, initial_num_channels=908, num_classes=2, num_channels=30, vocab_size = len(data.vocab),
input_size = data.max_len)
    trained_model = trainCNN(model, 20)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CNN model.')
    parser.add_argument('--pretrained', metavar='P', type=bool,
                    help='Whether or not to use pretrained embeddings')
    args = parser.parse_args()
    train_model()
