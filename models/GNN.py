import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import  scatter_




### Node embedding

class GNNLayer(MessagePassing): 
    def __init__(self, ndim):
        super(GNNLayer, self).__init__(aggr='add') 
        self.msg = nn.Linear(ndim*2, ndim*2)  
        self.msg_rev = nn.Linear(ndim*2, ndim*2) 
        self.upd = nn.GRUCell(2*ndim, ndim)
        
    def forward(self, edge_index, h):
        return self.propagate(edge_index, h=h)

    def message(self, h_j, h_i):
        m = torch.cat([h_j, h_i], dim=1)
        m, m_reverse = torch.split(m, m.size(0)//2, 0) 
        a = torch.cat([self.msg(m), self.msg_rev(m_reverse)], dim=0)
        return a 
    
    def update(self, aggr_out, h):
        h = self.upd(aggr_out, h)
        return h


class GNNinit(MessagePassing):   
    def forward(self, edge_index, h, h_prev):
        h_diff=self.propagate(edge_index, x=h_prev)
        h=h+h_diff
        return h,h_diff
    
        
    
    
class NodeEmb(nn.Module):
    def __init__(self, ndim, num_layers, num_node_atts, node_dropout, dropout):
        super(NodeEmb,self).__init__()
        
        self.ndim=ndim
        self.num_node_atts=num_node_atts
        self.num_layers = num_layers
        self.dropout = dropout
        self.Dropout = nn.Dropout(dropout)
        self.NodeInit = nn.Embedding(num_node_atts, ndim)
        self.GNNLayers = nn.ModuleList([GNNLayer(ndim) for _ in range(num_layers)])
        self.GNNinit = GNNinit()
         
        
        
    def forward(self, edge_index, node_atts):
        h = self.NodeInit(node_atts)
        edge_index = torch.cat([edge_index, torch.index_select(edge_index, 0, torch.tensor([1, 0]).to(h.device))], 1)   
        for layer in self.GNNLayers: 
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = layer(edge_index, h)
        return h
   


    
### Graph embedding

class GraphEmb(nn.Module):
    def __init__(self, ndim, gdim, aggr='gsum'):
        super(GraphEmb, self).__init__()
        self.ndim = ndim
        self.gdim = gdim
        self.aggr = aggr
        self.f_m = nn.Linear(ndim, gdim) 
        if aggr == 'gsum': 
            self.g_m = nn.Linear(ndim, 1) 
            self.sigm = nn.Sigmoid()

    def forward(self, h, batch):
        if self.aggr == 'mean':
            h = self.f_m(h)
            return scatter_('mean', h, batch)
        elif self.aggr == 'gsum':
            h_vG = self.f_m(h)  
            g_vG = self.sigm(self.g_m(h)) 
            h_G = torch.mul(h_vG, g_vG)   
            return scatter_('add', h_G, batch)
        

        
class GNNEncoder(nn.Module):
    def __init__(self, ndim, gdim, num_gnn_layers, num_node_atts, node_dropout=.0, g_aggr='gsum', dropout=.0):
        super().__init__()
        self.NodeEmb = NodeEmb(ndim, num_gnn_layers, num_node_atts, node_dropout, dropout)
        self.GraphEmb_mean = GraphEmb(ndim, gdim, g_aggr) 
        self.GraphEmb_var = GraphEmb(ndim, gdim, g_aggr)        
        
    def forward(self, edge_index, node_atts, batch):
        h = self.NodeEmb(edge_index, node_atts)
        h_G_mean = self.GraphEmb_mean(h, batch) 
        h_G_var = self.GraphEmb_var(h, batch) 
        return h_G_mean, h_G_var

    def number_of_parameters(self):
        return(sum(p.numel() for p in self.parameters() if p.requires_grad))  



    
# Accuracy Prediction

class GetAcc(nn.Module): 
    def __init__(self, gdim, dim_target, num_layers, dropout ):
        super(GetAcc, self).__init__()
        self.gdim = gdim
        self.num_layers = num_layers 
        self.dropout = dropout
        self.dim_target=dim_target
        self.lin_layers = nn.ModuleList([nn.Linear(gdim//(2**num), gdim//(2**(num+1))) for num in range(num_layers-1)]) 
        self.lin_layers.append(nn.Linear(gdim//(2**(num_layers-1)), dim_target))
        
    def forward(self, h):
        for layer in self.lin_layers[:-1]:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.relu(layer(h)) 
        h = self.lin_layers[-1](h)
        return h
    
    def __repr__(self):
        return '{}({}x Linear) Dropout(p={})'.format(self.__class__.__name__,
                                                self.num_layers,
                                                self.dropout
                                              )    

    
class GNNpred(nn.Module):
    def __init__(self, ndim, gdim, dim_target, num_gnn_layers, num_layers, num_node_atts, model_config):
        super().__init__()
        self.ndim= ndim
        self.gdim=gdim
        self.dim_target=dim_target
        self.num_gnn_layers=num_gnn_layers
        self.num_node_atts=num_node_atts
        self.config = model_config
        self.num_acc_layers=num_layers
        dropout=model_config['dropout_prob']
        g_aggr='gsum'
        node_dropout=.0
        self.NodeEmb = NodeEmb(ndim, num_gnn_layers, num_node_atts, node_dropout, dropout)
        self.GraphEmb_mean = GraphEmb(ndim, gdim, g_aggr)
        self.Accuracy = GetAcc(gdim, dim_target, num_layers, dropout=.0)
        
     
    def forward(self, graph_batch):
        node_atts, edge_index, batch = graph_batch.node_atts.long(), graph_batch.edge_index.long(), graph_batch.batch

        h = self.NodeEmb(edge_index, node_atts)
        h_G_mean = self.GraphEmb_mean(h, batch)

        acc = self.Accuracy(h_G_mean)
        return acc
