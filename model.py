import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, batch_size, weights, drop_prob=0.2, device = 'cpu'):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.embedding = nn.Embedding.from_pretrained(weights)

        self.gru1 = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.gru2 = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, sent1, sent2):
        h1_init = self.init_hidden(self.batch_size)

        sent1 = self.embedding(sent1)
        out1, h1 = self.gru1(sent1, h1_init)
        h1 = F.relu(h1)

        h2_init = self.init_hidden(self.batch_size)

        sent2 = self.embedding(sent2)
        out2, h2 = self.gru2(sent2, h2_init)
        h2 = F.relu(h2)

        h = torch.cat(h1,h2)

        out = self.fc1(h)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.out(out)
        out = nn.Sigmoid(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        return hidden