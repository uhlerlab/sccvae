import torch
import numpy as np
import argparse
import torch
import json

import sys
sys.path.append('./')

from causal.dataset import PerturbDataset, SCDATA_sampler
import causal.train
import os
from torch.utils.data import DataLoader
from datetime import datetime

def get_data(batch_size, nontargeting, ptb_type, ptb_dim, ood, finetune, marker, modes = None, ptbbatch = False, split_num = None, tiny = False, parent_dir = False):
    datasets, dataloaders, dataset_sizes = {}, {}, {}

    if modes is None:
        if nontargeting:
            modes = ['train']
        else:
            if finetune:
                modes = ['test']
            else:
                modes = ['train', 'val']

    prefix = '.' if parent_dir else ''
    for mode in modes:
        if ood:
            if not tiny:
                perturbfile = f'{prefix}./h5ad_datafiles/k562_annotated_raw_ood_split_{split_num}.h5ad'
            else:
                perturbfile = f'{prefix}./h5ad_datafiles/k562_annotated_raw_ood_split_{split_num}_tiny.h5ad'
        else:
            perturbfile = f'{prefix}./h5ad_datafiles/k562_annotated_raw_allnontargetinginval.h5ad'
        datasets[mode] = PerturbDataset(
            perturbfile = perturbfile,
            gene_id_file = f'{prefix}./train_util_files/new_estimated_dag_gene.csv',
            mode = mode if not nontargeting else 'non-targeting',
            ptb_type = ptb_type,
            ptb_dim = ptb_dim, 
            marker_genes = marker,
            parent_dir = parent_dir
        )

        if ptbbatch:
            dataloaders[mode] = DataLoader(
                datasets[mode],
                batch_sampler=SCDATA_sampler(datasets[mode], batch_size),
                num_workers=0
            )
        else:
            dataloaders[mode] = DataLoader(
                datasets[mode],
                batch_size=batch_size,
                shuffle=True,
                num_workers=0
            )
        dataset_sizes[mode] = len(datasets[mode])
    
    if finetune:
        dataset_sizes['train'] = dataset_sizes['test']
        dataloaders['train'] = dataloaders['test']
        datasets['train'] = datasets['test']

        del datasets['test']
        del dataloaders['test']
        del dataset_sizes['test']

    return datasets, dataloaders, dataset_sizes

def main(args, CONFIG):
    print(torch.cuda.is_available())
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    print(f'using device {device}')

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if not os.path.isdir(f'{args.out_dir}/outs'):
        os.mkdir(f'{args.out_dir}/outs')
    
    datasets, dataloaders, _ = get_data(
        batch_size=32, 
        nontargeting=args.n, 
        ptb_type = CONFIG['ptb_type'] if '_finetune' not in CONFIG['ptb_type'] else CONFIG['ptb_type'][:-9], 
        ptb_dim = CONFIG['ptb_dim'], 
        ood = CONFIG['ood'], 
        finetune = CONFIG['finetune'],
        marker = CONFIG['marker'],
        ptbbatch = CONFIG['ptbbatch'],
        split_num = CONFIG['split_num']
    )
    
    desc = args.description
    
    savedir = f'{args.out_dir}/outs/{desc}'
    print(f'saving to {savedir}')

    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    
    with open(f"{savedir}/config.json", 'w') as f: 
        json.dump(CONFIG, f)
    
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    _ = causal.train.train(
        device=device,
        dataloaders=dataloaders,
        datasets = datasets,
        learning_rate=CONFIG['learning_rate'],
        num_epochs=CONFIG['num_epochs'],
        beta_start=CONFIG['beta_start'],
        beta_max=CONFIG['beta_max'],
        gamma_max = CONFIG['gamma_max'],
        gamma_start = CONFIG['gamma_start'],
        savedir = savedir,
        hidden_size=CONFIG['hidden_size'],
        n_hidden=CONFIG['n_hidden'],
        beta_end = CONFIG['beta_end'],
        mode = CONFIG['mode'],
        schedule = CONFIG['schedule'],
        ptb_dim = CONFIG['ptb_dim'],
        pretrain_dir = CONFIG['pretrained_dir'],
        ptb_encode = CONFIG['ptb_type'],
        marker = CONFIG['marker'],
        train_hard = CONFIG['train_hard'],
        recon_scale = CONFIG['recon_scale']
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-o', '--out-dir', default='.', type=str,
                        help='name of save directory')
    parser.add_argument('-u', '--upload-dir', default='', type=str,
                        help='directory of pretrained model')
    parser.add_argument('-l', '--learning-rate', default=5e-4, type=float,
                        help='learning rate')
    parser.add_argument('-b', '--beta-max', default=1, type=float,
                        help='max beta')
    parser.add_argument('-j', '--gamma-start', default=0, type=int,
                        help='gamma')
    parser.add_argument('-g', '--gamma-max', default=10, type=float,
                        help='gamma')
    parser.add_argument('-a', '--beta-start', default=0, type=int,
                        help='start of beta schedule')
    parser.add_argument('--seed', default=20, type=int,
                        help='seed')
    parser.add_argument('--schedule', action='store_true', help='use a lr schedule or not')
    parser.add_argument('-p', '--ptb_type', default='pca', type=str,
                        help='Perturbation encoding type: [onehot, expression, genept]')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='cuda device number')
    parser.add_argument('-r', '--rate', default=0.1, type=float,
                        help='rate for lr scheduler')
    parser.add_argument('-t', '--patience', default=10, type=int,
                        help='patience for lr scheduler')
    parser.add_argument('--num-epochs', default=100, type=int,
                        help='number of training epochs')
    parser.add_argument('--perturb-dim', default=50, type=int,
                        help='perturbation dimension')
    parser.add_argument('--finetune', action='store_true',
                        help='True if finetuning OOD preds else false')
    parser.add_argument('-n', action='store_true', help = 'use nontargeting only')
    parser.add_argument('-m', default='full', type=str, help = 'causal, conditional, random')
    parser.add_argument('--ood', action='store_true', help = 'OOD evals or no')
    parser.add_argument('--marker', action='store_true', help = 'true if using top 50 marker genes')
    parser.add_argument('-e', action='store_true', help = 'run evaluations')
    parser.add_argument('--recon-scale', default=0.25, type=float, help='scale for mse X')
    parser.add_argument('-x', '--description', default='', type=str,
                        help='brief description of what this run includes')
    parser.add_argument('--n-hidden', default=2, type=int,
                        help='num hidden layers')
    parser.add_argument('--split-num', default=-1, type=int)
    parser.add_argument('--random-graph-seed', default=0, type=int, help='seed for random graph')
    
    torch.set_default_dtype(torch.float64)

    args = parser.parse_args()

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    sched = {
        'factor': args.rate,
        'patience': args.patience
    }

    CONFIG = {
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'beta_start': args.beta_start,
        'beta_max': args.beta_max,
        'gamma_max': args.gamma_max,
        'gamma_start': args.gamma_start,
        'n_hidden': args.n_hidden,
        'hidden_size': 1024,
        'beta_end': args.num_epochs, 
        'mode': args.m,
        'ptb_type': args.ptb_type,
        'schedule': sched if args.schedule else None,
        'pretrained_dir': args.upload_dir if args.upload_dir != '' else None,
        'eval': args.e,
        'ptb_dim': args.perturb_dim,
        'ood': args.ood,
        'finetune': args.finetune,
        'marker': args.marker,
        'train_hard': True,
        'ptbbatch': True,
        'split_num': args.split_num,
        'recon_scale': args.recon_scale,
        'rand_graph_seed': args.random_graph_seed
    }

    main(args, CONFIG)