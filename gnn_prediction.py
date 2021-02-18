import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import pdb
import numpy as np
import time 
import json
import glob
import os
import sys
import torch
import random
import torch.nn as nn
from torch_geometric.data import Data, DataLoader

from tqdm import tqdm
from models import  GNNpred

import utils


import argparse
parser = argparse.ArgumentParser(description='PerformancePrediciton')
parser.add_argument('--model', help='which surrogate model to fit', default='GNNpred')
parser.add_argument('--save_interval', type=int, default=50, help='how many epochs to wait to save model')
parser.add_argument('--log_dir', type=str, help='Experiment directory', default='experiments')
#parser.add_argument('--sample', action='store_true', default=False, help='if GNN trained on whole dataset')
parser.add_argument('--sample', action='store_true', default=True, help='if GNN trained on whole dataset')
parser.add_argument('--training_size', type=int, help='size of training data ', default='1000')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--gpu',type=int, default=2) 
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--test', action='store_true', default=False, help='if prediction on test acc of NAS-Bench-101')
#parser.add_argument('--dryrun', action='store_true', default=False, help='if prediction on test acc of NAS-Bench-101')
parser.add_argument('--dryrun', action='store_true', default=True, help='if prediction on test acc of NAS-Bench-101')

args = parser.parse_args()
if args.no_cuda:
    device = torch.device("cpu")
else:
    torch.cuda.set_device(args.gpu)    
    device = torch.device("cuda")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

#Create Log Directory
if args.sample:
    if args.test:
        log_dir= os.path.join(args.log_dir, '{}_test'.format(args.model),'{}/{}'.format( args.training_size,time.strftime("%Y%m%d-%H%M%S")))
    else:
        log_dir= os.path.join(args.log_dir, args.model,'{}/{}'.format( args.training_size,time.strftime("%Y%m%d-%H%M%S")))
else:
    if args.test:
        log_dir= os.path.join(args.log_dir, '{}_test'.format(args.model),'{}'.format(time.strftime("%Y%m%d-%H%M%S")))
    else:
        log_dir= os.path.join(args.log_dir, args.model,'{}'.format(time.strftime("%Y%m%d-%H%M%S")))

utils.create_exp_dir(log_dir, scripts_to_save=glob.glob('*.py'))


# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(log_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main(args):
    
    logging.info("args = %s", args)

    # Get configs
    config_path='model_configs.json'
    config = json.load(open(config_path, 'r'))

    # Load Training Data
    data_root=config['train_data_path']
    data=torch.load(data_root)

    # Sample Train data 
    if args.sample:
        np.random.seed(args.seed)
        random_shuffle = np.random.permutation(range(len(data)))
        sample_amount=args.training_size
        other_data=[data[i] for i in random_shuffle[sample_amount:]]
        train_data=[data[i] for i in random_shuffle[:sample_amount]]

    # Get Validation and Test Data
    val_data_root=config['val_data_path']
    val_data=torch.load(val_data_root)

    test_data_root=config['test_data_path']
    test_data=torch.load(test_data_root)

    #Load Model
    model = eval(args.model)(config['gnn_node_dimensions'], config['gnn_hidden_dimensions'], config['dim_target'], 
                            config['num_gnn_layers'], config['num_acc_layers'], config['num_node_atts'], 
                            model_config=config).to(device)



    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 
    criterion = nn.MSELoss()

    budget = config['epochs']
    for epoch in range(1, int(budget)+1):
        logging.info('epoch: %s', epoch)
       
        # training
        train_obj, train_results=train(train_data, model, criterion, optimizer, config, epoch, device)
        

        if epoch % args.save_interval == 0:
            logger.info('save model checkpoint {}  '.format(epoch))
            model_name = os.path.join(log_dir, 'model_checkpoint{}.obj'.format(epoch))
            torch.save(model.state_dict(), model_name)
            optimizer_name = os.path.join(log_dir, 'optimizer_checkpoint{}.obj'.format(epoch))
            torch.save(optimizer.state_dict(), optimizer_name)


        # validation    
        valid_obj, valid_results = infer(val_data, model, criterion, config, epoch, device )
            
            
        # testing
        test_obj, test_results= test(test_data, model, criterion, config,epoch, device)


        config_dict = {
            'epoch': epoch,
            'loss': train_results["rmse"],
            'val_rmse': valid_results['rmse'],
            'test_rmse': test_results['rmse'],
            'test_mse': test_results['mse'],
            }

        # Save the entire model
        if epoch % args.save_interval == 0:
            logger.info('save model checkpoint {}  '.format(epoch))
            filepath = os.path.join(log_dir, 'model_{}.obj'.format(epoch))
            torch.save(model.state_dict(), filepath)
        

        with open(os.path.join(log_dir, 'results.txt'), 'a') as file:
                json.dump(str(config_dict), file)
                file.write('\n')

        
        if args.dryrun:
            break


def train(train_data, model, criterion, optimizer, config, epoch, device):
    objs = utils.AvgrageMeter()
    # TRAINING
    preds = []
    targets = []
        
    model.train()
    data_loader = DataLoader( train_data, shuffle=True, num_workers=config['num_workers'], pin_memory=True, batch_size=config['batch_size'])
    for step, graph_batch in enumerate(data_loader):
        graph_batch = graph_batch.to(device)
        pred = model(graph_batch=graph_batch).view(-1)
        if args.test:
            loss = criterion(pred, (graph_batch.test_acc))
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch.test_acc.detach().cpu().numpy())
        else:
            loss = criterion(pred, (graph_batch.acc))
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch.acc.detach().cpu().numpy())

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = graph_batch.num_graphs
        objs.update(loss.data.item(), n)
        if args.dryrun:
            break

    logging.info('train %03d %.5f', step, objs.avg)
    train_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
    logging.info('train metrics:  %s', train_results)
    return objs.avg, train_results

def infer(val_data, model, criterion, config, epoch, device):
    objs = utils.AvgrageMeter()

    # VALIDATION
    preds = []
    targets = []

    model.eval()
    data_loader = DataLoader( val_data, shuffle=False, num_workers=config['num_workers'], batch_size=config['batch_size'])
    for step, graph_batch in enumerate(data_loader):
        graph_batch = graph_batch.to(device)
        pred = model(graph_batch=graph_batch).view(-1)             
        if args.test:
            loss = criterion(pred, (graph_batch.test_acc))
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch.test_acc.detach().cpu().numpy())
        else:
            loss = criterion(pred, (graph_batch.acc))
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch.acc.detach().cpu().numpy())
            
        n = graph_batch.num_graphs
        objs.update(loss.data.item(), n)
        if args.dryrun:
            break

    logging.info('valid %03d %.5f', step, objs.avg)

    val_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
    logging.info('val metrics:  %s', val_results)
    return objs.avg, val_results

def test(test_data, model, criterion, config, epoch, device):
    objs = utils.AvgrageMeter()
    preds = []
    targets = []
    model.eval()
    data_loader = DataLoader(test_data, shuffle=False, num_workers=config['num_workers'], batch_size=config['batch_size'])
    for step, graph_batch in enumerate(data_loader):
        graph_batch = graph_batch.to(device)
        pred = model(graph_batch=graph_batch).view(-1)
        if args.test:
            loss = criterion(pred, (graph_batch.test_acc))
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch.test_acc.detach().cpu().numpy())
        else:
            loss = criterion(pred, (graph_batch.acc))
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch.acc.detach().cpu().numpy())
            
        n = graph_batch.num_graphs
        objs.update(loss.data.item(), n)
        if args.dryrun:
            break

    logging.info('test %03d %.5f', step, objs.avg)
    test_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
    logging.info('test metrics %s', test_results)
    return objs.avg, test_results


    
if __name__ == '__main__':
    main(args)
    
