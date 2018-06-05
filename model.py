import torch
import torch.nn as nn
import torch.nn.functional as F


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_topics, dropout):
        super().__init__()
        self.drop1 = nn.Dropout(dropout)
        # encoder hidden layers
        self.fc1 = nn.Linear(vocab_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # for latent code
        self.fcmu = nn.Linear(hidden_size, num_topics)
        self.fclv = nn.Linear(hidden_size, num_topics)
        self.bnmu = nn.BatchNorm1d(num_topics)
        self.bnlv = nn.BatchNorm1d(num_topics)
        # decoder 
        self.fc3 = nn.Linear(num_topics, vocab_size)
        self.bn  = nn.BatchNorm1d(vocab_size)
        self.drop2 = nn.Dropout(dropout)
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)
    
    def forward(self, inputs):
        h1 = F.softplus(self.fc1(inputs))
        h2 = F.softplus(self.fc2(h1))
        h = self.drop1(h2)
        mu = self.bnmu(self.fcmu(h))
        lv = self.bnlv(self.fclv(h))
        z = self.reparameterize(mu, lv)
        p = self.drop2(F.softmax(z, dim=1))
        outputs = F.softmax(self.bn(self.fc3(p)), dim=1)
        return outputs, mu, lv
