from torch.utils.data import Dataset

from torch.utils.data import Dataset

class DebatesSets(DebatesSets):
    def __init__(self,sent1,sent2, y, max_lens):
        self.y = y

        #pad sequences up to max_len
        def pad_sents(sentences):
            padded_vectors = []
            for sentence in sentences:
                seq = np.zeros(max_len)
                i = 0
                for wordindex in sentence:
                    seq[i] = wordindex
                    i += 1
                    if i == len(sentence):
                        padded_vectors.append(np.copy(seq))
            return padded_vectors
        
        self.sent1 = pad_sents(sent1)
        self.sent2 = pad_sents(sent2)
        
        
    def __getitem__(self, index):

        sent1 = self.sent1[index]
        sent2 = self.sent2[index]
        y = self.y[index]
        
        return (sent1,sent2), y


    def __len__(self):
        return len(self.sent1)

# class DebatesSets(Dataset):
#     def __init__(self,sent1,sent2, y):
#         self.sent1 = sent1
#         self.sent2 = sent2
#         self.y = y

#     def __getitem__(self, index):

#         sent1 = self.sent1[index]
#         sent2 = self.sent2[index]
#         y = self.y[index]

#         return (torch.LongTensor(sent1),torch.LongTensor(sent2)), y

#     def __len__(self):
#         return len(self.sent1)


# class DebatesSets(Dataset):
#     def __init__(self,sent1,sent2, y):
#         self.sent1 = sent1
#         self.sent2 = sent2
#         self.y = y

#     def __getitem__(self, index):

#         sent1 = self.sent1[index]
#         sent2 = self.sent2[index]
#         y = self.y[index]

#         return (torch.LongTensor(sent1),torch.LongTensor(sent2)), y

#     def __len__(self):
#         return len(self.sent1)
