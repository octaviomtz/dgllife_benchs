#%%
import numpy as np
import torch
import torch.nn as nn

from dgllife.model import load_pretrained
from dgllife.utils import smiles_to_bigraph, EarlyStopping, Meter
from functools import partial
from torch.optim import Adam
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from utils_load import load_dataset
from utils import init_featurizer, mkdir_p, split_dataset, get_configure
from utils import collate_molgraphs, load_model, predict
from rdkit import Chem
from rdkit.Chem import Draw
from itertools import islice
from pprint import pprint

# %%
def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        if len(smiles) == 1:
            # Avoid potential issues with batch normalization
            continue

        labels, masks = labels.to(args['device']), masks.to(args['device'])
        logits = predict(args, model, bg)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter.update(logits, labels, masks)
        if batch_id % args['print_every'] == 0:
            print('epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}'.format(
                epoch + 1, args['num_epochs'], batch_id + 1, len(data_loader), loss.item()))
    train_score = np.mean(train_meter.compute_metric(args['metric']))
    print('epoch {:d}/{:d}, training {} {:.4f}'.format(
        epoch + 1, args['num_epochs'], args['metric'], train_score))
# %%
def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(args['device'])
            logits = predict(args, model, bg)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.compute_metric(args['metric']))

#%%
def main(args, exp_config, train_set, val_set, test_set):
    print(args)
    if args['featurizer_type'] != 'pre_train':
        exp_config['in_node_feats'] = args['node_featurizer'].feat_size()
        if args['edge_featurizer'] is not None:
            exp_config['in_edge_feats'] = args['edge_featurizer'].feat_size()
    exp_config.update({
        'n_tasks': args['n_tasks'],
        'model': args['model']
    })

    train_loader = DataLoader(dataset=train_set, batch_size=exp_config['batch_size'], shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    val_loader = DataLoader(dataset=val_set, batch_size=exp_config['batch_size'],
                            collate_fn=collate_molgraphs, num_workers=args['num_workers'])
    test_loader = DataLoader(dataset=test_set, batch_size=exp_config['batch_size'],
                             collate_fn=collate_molgraphs, num_workers=args['num_workers'])

    if args['pretrain']:
        args['num_epochs'] = 0
        if args['featurizer_type'] == 'pre_train':
            model = load_pretrained('{}_{}'.format(
                args['model'], args['dataset'])).to(args['device'])
        else:
            model = load_pretrained('{}_{}_{}'.format(
                args['model'], args['featurizer_type'], args['dataset'])).to(args['device'])
    else:
        model = load_model(exp_config).to(args['device'])
        loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
        optimizer = Adam(model.parameters(), lr=exp_config['lr'],
                         weight_decay=exp_config['weight_decay'])
        stopper = EarlyStopping(patience=exp_config['patience'],
                                filename=args['result_path'] + '/model.pth',
                                metric=args['metric'])

    for epoch in range(args['num_epochs']):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['metric'],
            val_score, args['metric'], stopper.best_score))

        if early_stop:
            break

    if not args['pretrain']:
        stopper.load_checkpoint(model)
    val_score = run_an_eval_epoch(args, model, val_loader)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print('val {} {:.4f}'.format(args['metric'], val_score))
    print('test {} {:.4f}'.format(args['metric'], test_score))

    with open(args['result_path'] + '/eval.txt', 'w') as f:
        if not args['pretrain']:
            f.write('Best val {}: {}\n'.format(args['metric'], stopper.best_score))
        f.write('Val {}: {}\n'.format(args['metric'], val_score))
        f.write('Test {}: {}\n'.format(args['metric'], test_score))

#%%
def get_args():
    args = dict()
    args['dataset']= 'MUV'
    args['model']= 'GCN'
    args['featurizer_type']= 'canonical' #'attentivefp'
    args['pretrain']= False
    args['split']= 'scaffold'
    args['split_ratio']= '0.8,0.1,0.1'
    args['metric']= 'roc_auc_score'
    args['num_epochs']= 1000
    args['num_workers']= 0
    args['print_every']= 20
    args['result_path']= 'classification_results'
    args = init_featurizer(args)
    args['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return args
#%%
args = get_args()
dataset = load_dataset(args)                     
args['n_tasks'] = dataset.n_tasks
mkdir_p(args['result_path'])
train_set, val_set, test_set = split_dataset(args, dataset)

# %% OVERVIEW
df = train_set.dataset.df
print(f'df = {df.shape}')
print(f'n_tasks = {dataset.n_tasks}')
for i in df.columns[:-1]:
    arr = df[i].values
    print(f'{i, np.unique(arr[~np.isnan(arr)]), np.sum(arr==1)}, imbalance={np.sum(arr==1)/(np.sum(arr==1)+np.sum(arr==0)):.5f}')
molecules = [Chem.MolFromSmiles(smiles) for smiles in islice(train_set.dataset.smiles, 6)]
df.head()
Draw.MolsToGridImage(molecules)

# %%
exp_config = get_configure(args['model'], args['featurizer_type'], args['dataset'])

# %%
main(args, exp_config, train_set, val_set, test_set)

# %%
model = load_model(exp_config)
model



#%%
train_loader_temp = DataLoader(dataset=train_set, batch_size=8, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=args['num_workers'])
batch_data = next(iter(train_loader_temp))
smiles, bg, labels, masks = batch_data

# %%
from pprint import pprint
pprint(vars(bg))

#%%
batch_data = next(iter(train_loader_temp))
smiles, bg, labels, masks = batch_data

# %%
import matplotlib.pyplot as plt
print(bg.ndata.get('h').shape)
plt.imshow(bg.ndata.get('h').numpy())
# %%

# %%
