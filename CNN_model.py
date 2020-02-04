import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):       
    '''The current CNN structure with the convnet is largely borrowed from Rao & MacMahan's
    Natural Language Processing with PyTorch, 2019, pp 96. It has been modified to create one feature map
    by concatenating the outputs from two separate convolutions, one per each sentence
    '''
    def __init__(self, batch_size, initial_num_channels, num_classes, num_channels, weights):
        super(CNNClassifier, self).__init__()

        self.batch_size = batch_size
        self.convnet = nn.Sequential( nn.Conv1d(in_channels=initial_num_channels,
        self.num_classes = num_classes
        self.emb = nn.Embedding.from_pretrained(weights, freeze=True)

        out_channels=num_channels, kernel_size=2),
        nn.ELU(),
        nn.Conv1d(in_channels=num_channels,out_channels=num_channels,
        kernel_size=2, stride=2),
        nn.ELU(),
        nn.Conv1d(in_channels=num_channels,out_channels=num_channels,
        kernel_size=2, stride=2),
        nn.ELU(),
        nn.Conv1d(in_channels=num_channels,out_channels=num_channels,
        kernel_size=2),
        nn.ELU() )

    def set_dev(self, dev):
        self.dev = dev        
        
    def forward(self, sent1, sent2, apply_softmax=False):
        
        # perform separate convolutions
        features = []    
        for sent in [sent1, sent2]:
            sent = sent.to(dev)
            feat = self.emb(sent)
            feat = feat.transpose(1,2) 
            feat = self.convnet(feat).squeeze(dim=2) 
            feat = feat.view(self.batch_size, -1)
            features.append(feat)
        
        # concatenate the two feature maps into one
        features = torch.cat((features[0],features[1]),1)

        fc = nn.Linear(features.shape[1], self.num_classes)

        # we shouldn't send things back and forth between gpu and cpu – will this be fixed with data_loader?
        features = features.to('cpu')
        return F.softmax(features, dim=1)
