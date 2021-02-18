import logging
logging.basicConfig(format="%(asctime)s %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
import time 
import json
import glob
import os
import sys
import random

from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
#from torch.utils.data import DataLoader as BatchLoader


from models import  RNNEncoder, GetAcc

import utils


import argparse
parser = argparse.ArgumentParser(description='PerformancePrediciton_Baselines')
parser.add_argument('--model', choices=['RNN', 'RF_WD','RF_one', 'MLP_one'], help='which baseline predictor model to fit', default='RNN')
parser.add_argument('--save_interval', type=int, default=50, help='how many epochs to wait to save model')
parser.add_argument('--log_dir', type=str, help='Experiment directory', default='experiments')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--gpu',type=int, default=2)
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--dryrun', action='store_true', default=False, help='if prediction on test acc of NAS-Bench-101')


args = parser.parse_args()
if args.no_cuda:
    device = torch.device("cpu")
else:
    torch.cuda.set_device(args.gpu)    
    device = torch.device("cuda")
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

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
    
    #Get configs
    config_path='model_configs.json'
    config = json.load(open(config_path, 'r'))
    config['learning_rate']=args.learning_rate

    #Load Training data
    train_data=config['train_data_path']
    if args.model=='RNN':
        train_dataset = torch.load(train_data)
        train_dataset= utils.data_for_RNN(train_dataset)
    elif args.model=='RF_WD':
        train_data=config['train_data_rf_path']
        x_train, y_train=utils.prep_rf_wd(train_data)
        
    elif args.model=='RF_one': 
        train_dataset= torch.load(train_data)
        x_train, y_train=utils.make_one_hot(train_dataset)
        x_train=x_train.numpy()
        y_train=y_train.numpy()
        
    elif args.model=='MLP_one':
        train_dataset= torch.load(train_data)
        train_data, train_target_data=utils.make_one_hot(train_dataset)
        train_dataset = torch.cat([train_data, train_target_data.unsqueeze(1)], 1)
        
    else:
        raise NotImplementedError('Unknown gnn_type.') 

   
  
    #Load Validation Data
    validation_data=config['val_data_path']
    if args.model=='RNN':
        val_dataset = torch.load(validation_data)
        val_dataset= utils.data_for_RNN(val_dataset)
    elif args.model=='RF_WD':        
        val_data=config['val_data_rf_path']
        x_test, y_test=utils.prep_rf_wd(val_data)
        x_test = [x for _,x in sorted(zip(y_test,x_test))]
        y_test = [x for x in sorted(y_test)]       
    
    elif args.model=='RF_one' :
        test_dataset = torch.load(validation_data)
        x_test, y_test=utils.make_one_hot(test_dataset)
        x_test=x_test.numpy()
        y_test=y_test.numpy()
        
    elif args.model=='MLP_one':
        test_dataset = torch.load(validation_data)
        test_data, test_target_data=utils.make_one_hot(test_dataset)
        val_dataset = torch.cat([test_data, test_target_data.unsqueeze(1)], 1)

    else:
        raise NotImplementedError('Unknown gnn_type.') 
       
    #Load Model
  
    if args.model=='RNN':
        model = RNNEncoder(56, model_config=config).to(device)
        
    elif args.model=='MLP_one':
        model = GetAcc(config['one_hot_encoded'], config['dim_target'], config['num_acc_layers'], 
                           dropout=.0).to(device)
    
    criterion = nn.MSELoss()
    if args.model=='RNN' or args.model=='MLP_one':           
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) 
        budget = config['epochs']
        for epoch in range(1, int(budget)+1):
            logging.info('epoch: %s', epoch)

            # training
            train_obj, train_results=train(train_dataset, model, criterion, optimizer, config, epoch, device)

    #       validation    
            valid_obj, valid_results = infer(val_dataset, model, criterion, config, epoch, device)


            config_dict = {
                'epoch': epoch,
                'loss': train_results["rmse"],
                'val_rmse': valid_results['rmse'],
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
                      
    elif args.model=='RF_one' or args.model=='RF_WD':
        val_rmse=[]
        rf = RandomForestRegressor(n_estimators=10)
        rf.fit(x_train, np.ravel(y_train))
        y_hat = rf.predict(x_test)
        rmse=mean_squared_error(y_test, y_hat)**0.5
        val_rmse.append(rmse)
        logger.info('RMSE on the test set {}'.format(np.round(rmse, 6)))
                
        config_dict = {
                'val_rmse':rmse,
                } 

        with open(os.path.join(log_dir, 'results.txt'), 'a') as file:
                    json.dump(str(config_dict), file)
                    file.write('\n')

def train(train_dataset, model, criterion, optimizer, config, epoch, device):
    objs = utils.AvgrageMeter()
    
    # TRAINING
    preds = []
    targets = []
        
    model.train()
    train_loader = DataLoader(train_dataset, shuffle=True, num_workers=config['num_workers'], pin_memory=True, batch_size=config['batch_size'])
    if  model.__class__.__name__=='RNNEncoder':      
        for step, graph_batch in enumerate(train_loader):
            graph_batch = graph_batch.to(device)
            pred = model(graph_batch=graph_batch).view(-1)
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

    elif  model.__class__.__name__=='GetAcc':
        for step, graph_batch in enumerate(train_loader):
            graph_batch=graph_batch.to(device)
            pred=model(graph_batch[:,:-1]).view(-1)
            loss = criterion(pred.view(-1), (graph_batch[:,-1]).view(-1))
            
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch[:,-1].detach().cpu().numpy())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n = graph_batch.shape[0]
            objs.update(loss.data.item(), n)

            if args.dryrun:
                break
    
    logging.info('train %03d %.5f', step, objs.avg)
    train_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
    logging.info('train metrics:  %s', train_results)
    return objs.avg, train_results


def infer(val_dataset, model, criterion, config, epoch, device):
    objs = utils.AvgrageMeter()

    # VALIDATION
    preds = []
    targets = []

    model.eval()
    val_loader = DataLoader(val_dataset, shuffle=False, num_workers=config['num_workers'], batch_size=config['batch_size'])
    if model.__class__.__name__=='RNNEncoder' :
        for step, graph_batch in enumerate(val_loader):
            graph_batch = graph_batch.to(device)
            pred = model(graph_batch=graph_batch).view(-1)
            loss = criterion(pred, (graph_batch.acc))

            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch.acc.detach().cpu().numpy())
            n = graph_batch.num_graphs
            objs.update(loss.data.item(), n)
            if args.dryrun:
                break

            
    elif model.__class__.__name__=='GetAcc':
        for step, graph_batch in enumerate(val_loader):
            graph_batch=graph_batch.to(device)
            pred=model(graph_batch[:,:-1]).view(-1)
            loss = criterion(pred.view(-1), (graph_batch[:,-1]).view(-1))
            
            
            preds.extend((pred.detach().cpu().numpy()))
            targets.extend(graph_batch[:,-1].detach().cpu().numpy())

            n = graph_batch.shape[0]
            objs.update(loss.data.item(), n)
            if args.dryrun:
                break

    logging.info('valid %03d %.5f', step, objs.avg)
    val_results = utils.evaluate_metrics(np.array(targets), np.array(preds), prediction_is_first_arg=False)
    logging.info('val metrics:  %s', val_results)
    return objs.avg, val_results   



    
if __name__ == '__main__':
    main(args)
    
