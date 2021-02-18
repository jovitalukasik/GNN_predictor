import numpy as np
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
from scipy.stats import norm, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, r2_score
import shutil
import torch
import sys
import os
import pickle
import gzip
import collections
from collections import Counter

from tqdm import tqdm
import torch
from torch_geometric.data import Data, DataLoader

 

'''load and save objects'''
def save_object(obj, filename):
    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()

def load_object(filename):
    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()
    return ret

def load_module_state(model, state_name):
    pretrained_dict = torch.load(state_name)
    model_dict = model.state_dict()

    # to delete, to correct grud names
    '''
    new_dict = {}
    for k, v in pretrained_dict.items():
        if k.startswith('grud_forward'):
            new_dict['grud'+k[12:]] = v
        else:
            new_dict[k] = v
    pretrained_dict = new_dict
    '''

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)
    return

def evaluate_metrics(y_true, y_pred, prediction_is_first_arg):
    """
    Create a dict with all evaluation metrics
    """

    if prediction_is_first_arg:
        y_true, y_pred = y_pred, y_true

    metrics_dict = dict()
    metrics_dict["mse"] = mean_squared_error(y_true, y_pred)
    metrics_dict["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
#     metrics_dict["r2"] = r2_score(y_true, y_pred)
    metrics_dict["kendall_tau"], p_val = kendalltau(y_true, y_pred)
    metrics_dict["spearmanr"] = spearmanr(y_true, y_pred).correlation

    return metrics_dict

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)
            
            
def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def data_for_RNN(dataset):    
    data_list=[]
    for graph in dataset:
        adjacency=make_one_hot_nodes(graph.node_atts)
        data = Data(edge_index=torch.LongTensor(graph.edge_index),
    #             edit=torch.tensor([graph.edit]),
                node_atts=torch.LongTensor(graph.node_atts),
                num_nodes = graph.node_atts.size(0),
                acc = torch.tensor([graph.acc]),
                adjacency= torch.tensor([adjacency]),
#                 test_acc=torch.tensor([graph.test_acc]),
#                 training_time=torch.tensor([graph.training_time])
                   )  
        data_list.append(data)    
    return data_list

def make_one_hot_nodes(nodes):
    make_one_hot = lambda index: torch.eye(5)[index.view(-1).long()]
    emb_len=7

    l = make_one_hot(torch.LongTensor(nodes)).transpose(0,1)
    lp=torch.stack([torch.cat([i, i.new_zeros(emb_len - i.size(0))], 0) for i in l],1).view(-1).numpy()
    return lp

def make_one_hot(graph_data):
    nodes=[]
    target=torch.Tensor()
    for graph in tqdm(graph_data):
        node_attr = graph.node_atts.numpy()
        nodes.append(node_attr)
        val_acc = graph.acc
        target=torch.cat([target,val_acc])
        
    make_one_hot = lambda index: torch.eye(5)[index.view(-1).long()]
    emb_len=7
    x=torch.Tensor()
    for i in tqdm(nodes):
        l = make_one_hot(torch.LongTensor(i)).transpose(0,1)
        lp=torch.stack([torch.cat([i, i.new_zeros(emb_len - i.size(0))], 0) for i in l],1).view(-1).reshape(1,-1)
        x=torch.cat([x, lp])

    return x, target

    

def get_depth(a, b, c):
    g = {}

    inp = [x for x in a.split(' ')]
    out = [x for x in b.split(' ')]
    typ = [x for x in c.split(' ')]

    start = str(typ.index('1'))
    end = str(typ.index('0'))

    for j in range(len(inp)):
        if inp[j] not in g:
            g[inp[j]] = []
        g[inp[j]].append(out[j])

    # print(g, start, end)

    bre = max([len(g[x]) for x in g.keys()])
    dep = 0
    ex = []
    q = [[start]]
    found = False
    while q and not found:
        p = q.pop()
        v = p[-1]
        if p not in ex:
            nei = g[v]
            
            for n in nei:
                newP=list(p)
                newP.append(n)
                if n == end:
                    dep = len(newP)
                    found = True
                    break
                else:
                    q.append(newP)
            ex.append(v)
    return bre, dep


def to_feat(lines):
    cnts = [0,0,0,0,0,0,0]
    performance = float(lines[-1])
    c = Counter([int(x) for x in lines[0].split(' ')])
    # edges = 0
    for i in c.most_common():
        cnts[i[0]] = i[1]
    empty = [1,1,1,1,1,1,1]
    types = []
    for i in range(5):
        types.append([0,0,0,0,0,0,0])
    for i, e in enumerate(lines[-2].split(' ')):
        empty[i] = 0
        types[int(e)][i] = 1
    bre, dep = get_depth(lines[0], lines[1], lines[2])
    return cnts + types[0] + types[1] + types[2] + types[3] + types[4] + [dep], performance

def to_feat2(lines):
    # cnts = [0,0,0,0,0,0,0]
    matrix = []
    for i in range(7):
        row = []
        for j in range(7):
            row.append(0)
        matrix.append(row)
    for i in lines[0].split(' '):
        for j in lines[1].split(' '):
            matrix[int(i)][int(j)] = 1
    
    performance = float(lines[-1])
    c = Counter([int(x) for x in lines[0].split(' ')])
    # edges = 0
    # for i in c.most_common():
    #     cnts[i[0]] = i[1]
    empty = [1,1,1,1,1,1,1]
    types = []
    for i in range(5):
        types.append([0,0,0,0,0,0,0])
    for i, e in enumerate(lines[-2].split(' ')):
        empty[i] = 0
        types[int(e)][i] = 1
    mm = []
    for r in matrix:
        mm = mm + r
    return mm + types[0] + types[1] + types[2] + types[3] + types[4], performance


def prep_rf_wd(data):
    x_data = []
    y_data = []

    with open(data, 'r') as f:
        i = 0
        block = []
        for line in f:
            block.append(line.strip())
            i += 1
            if i == 4:
                i = 0
                x, y = to_feat(block)
                x_data.append(x)
                y_data.append(y)
                block = []
        
    logger.info('amount training data {}(input) {}(target)'.format(len(x_data), len(y_data))) 
        
    return x_data, y_data