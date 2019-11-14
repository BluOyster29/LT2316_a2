from torch.utils.data import Dataset

class DebatesSets(Dataset):
    def __init__(self,sent1,sent2, y):
        self.sent1 = sent1
        self.sent2 = sent2
        self.y = y

    def __getitem__(self, index):

        sent1 = self.sent1[index]
        sent2 = self.sent2[index]
        y = self.y[index]

        return (torch.LongTensor(sent1),torch.LongTensor(sent2)), y

    def __len__(self):
        return len(self.sent1)
