import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import  Linear, ReLU




class RNNEncoder(torch.nn.Module):
    def __init__(self, sdim, model_config, layers=1, mlp_dropout=0):
        super(RNNEncoder, self).__init__()
        
        
        self.sdim=sdim
        sdim=20
        self.config=model_config
        self.layers=layers
        dropout = model_config['dropout_prob']
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(5,20)
        self.mlp_dropout=mlp_dropout
        
        self.rnn = nn.LSTM(sdim, sdim, layers, batch_first=True, dropout=dropout)
        
        self.lin_layers = nn.ModuleList([nn.Linear(sdim//(2**num), sdim//(2**(num+1))) for num in range(model_config['num_acc_layers']-1)]) 
        self.lin_layers.append(nn.Linear(sdim//(2**(model_config['num_acc_layers']-1)), 1))
  
        
        self.sigm = nn.Sigmoid()

       
     

    def forward(self,graph_batch):
        x= graph_batch.adjacency.long()

        
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)

        out, hidden = self.rnn(embedded)
        # RNN in PNAS does not use normalization
        out = F.normalize(out, 2, dim=1)

        
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=1)
        for layer in self.lin_layers[:-1]:
            out = F.relu(layer(out))             
#             pdb.set_trace()
        h = self.lin_layers[-1](out)
        acc = self.sigm(h)
        return acc

            


    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))
    

    
