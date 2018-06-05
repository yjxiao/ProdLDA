import os
import pickle
import numpy as np
import torch


class Corpus:
    def __init__(self, datadir):
        filenames = ['train.txt.npy', 'test.txt.npy']
        self.datapaths = [os.path.join(datadir, x) for x in filenames]
        with open(os.path.join(datadir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
        self.train, self.test = [
            Data(dp, len(self.vocab)) for dp in self.datapaths]


class Data:
    def __init__(self, datapath, vocab_size):
        data = np.load(datapath, encoding='bytes')
        self.data = np.array([
            np.bincount(x.astype('int'), minlength=vocab_size)\
            for x in data if np.sum(x)>0])
        
    @property
    def size(self):
        return len(self.data)
    
    def get_batch(self, batch_size, start_id=None):
        if start_id is None:
            batch_idx = np.random.choice(np.arange(self.size), batch_size)
        else:
            batch_idx = np.arange(start_id, start_id + batch_size)
        batch_data = self.data[batch_idx]
        data_tensor = torch.from_numpy(batch_data).float()
        return data_tensor
