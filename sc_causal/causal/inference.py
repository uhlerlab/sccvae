import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader


import sys
sys.path.append('./')


from causal.model import CausalVAE_Gaussian as CausalVAE
# from model import CausalVAE_NB_Paired # as CausalVAE

from torch.distributions import Normal
from tqdm import tqdm_pandas, tqdm
from causal.dataset import PerturbDataset, SCDATA_sampler, PairedPerturbDataset
import argparse

BATCHSIZE = 32

def get_eval_data(batch_size, data_mode, ptb_type, ptb_dim, split = -1):
    if split == -1:
        ptbfile = f'./h5ad_datafiles/k562_annotated_raw_allnontargetinginval.h5ad'
    else:
        ptbfile = f'./h5ad_datafiles/k562_annotated_raw_ood_split_{split}.h5ad'

    dataset = PerturbDataset(
            perturbfile = ptbfile,
            gene_id_file = './train_util_files/new_estimated_dag_gene.csv',
            mode = data_mode,
            ptb_type=ptb_type,
            ptb_dim = ptb_dim
    )
    dataloader = DataLoader(
            dataset,
            batch_sampler=SCDATA_sampler(dataset, batch_size),
            num_workers=0
    )
    return dataset, dataloader

# If ptb is specified, only evaluates for the given ptbs.
def compute_predictions(
        model_path, 
        inference_modes, 
        device, 
        ptb=None, 
        variable='y', 
        setting = 'causal', 
        ood=False, 
        downsample = 1, 
        ptb_type = 'onehot', 
        ptb_dim = 512, 
        data_mode = None, 
        shiftval = None, 
        data = None,
        normed = True,
        hard = True,
        ood_split = -1,
        rand_graph_seed = 0,
        net = None):
    print(device)
    print(inference_modes)

    if data_mode is None:
        if ptb is None or 'non-targeting' not in ptb:
            data_mode = 'test'
        else:
            data_mode = 'train'
    
    print(data_mode)
    print(ptb)

    if data is None:
        dataset, dataloader = get_eval_data(
            batch_size=BATCHSIZE,
            data_mode=data_mode, 
            ptb_type=ptb_type,
            ptb_dim = ptb_dim,
            split = ood_split
        )
    else:
        dataset, dataloader = data

    if net is None:
        model = CausalVAE(
            ptb_dim=ptb_dim,
            exp_dim = 8563,
            z_dim=512,
            enc_hidden = [1024, 1024], 
            dec_hidden = [1024, 1024], 
            B_filename='./train_util_files/B_512_upper_triangular.npy',
            device = device,
            mode = setting,
            ptb_encode = ptb_type,
            rand_graph_seed=rand_graph_seed
        )
    else:
        model = net


    state_dict = torch.load(model_path)['model_state']

    model.load_state_dict(state_dict)
    
    model = model.to(device)

    def compute_for_shift(shiftval):

        bar = tqdm(dataloader, desc='computing predictions: '.ljust(20))

        ads_true = []
        likelihood = {}


        variables = {
            'y': [],
            'p': [],
            'p_input': [],
            'z': [],
            'u': [],
            'c': [],
            'shift': [],
            'z_shift': [],
            'u_shift': [],
            'z_mean': [],
            'ptb_name': []
        }

        total = 0
        exists_nontargeting = False

        for i, batch in enumerate(bar):
            total += 1
            y, p, p1h, gene = batch
            gene = gene[0]

            if ptb is not None and gene != 'non-targeting' and gene not in ptb:
                continue
            
            y, p, p1h = y.to(device), p.to(device), p1h.to(device)

            adata = ad.AnnData(y.detach().cpu().numpy())

            # if data_mode != 'train' and gene == 'non-targeting':
            if ptb is not None and 'non-targeting' not in ptb and gene == 'non-targeting':
                adata.obs['y'] = ['nontargeting distribution']*y.shape[0]
                exists_nontargeting = True
            else:
                adata.obs['y'] = ['ground truth']*y.shape[0]
            adata.obs['ptb'] = [gene]*y.shape[0]
            adata.obs['mode'] = ['ref']*y.shape[0]
            adata.obs['idx'] = [i] * y.shape[0]
            ads_true.append(adata)

            # if data_mode != 'train' and gene == 'non-targeting':
            if ptb is not None and 'non-targeting' not in ptb and gene == 'non-targeting':
                continue # if not evaluating on train set, just use nontargeting as ref and don't predict on it

            for mode in inference_modes:
                pred_vars = inference(model, p, p1h, device, shiftval = shiftval, gene= gene, normed = normed, hard = hard)

                for var in pred_vars:
                    if pred_vars[var] is not None:
                        pred_vars[var] = pred_vars[var].detach().cpu().numpy()
                
                variables['ptb_name'].append(gene)

                for var in variables:
                    if var in ['lkl', 'rates', 'dropouts', 'lib_vars', 'ptb_name']:#, 'shift']:
                        continue
                    pred = pred_vars[var]
                    pred = pred.reshape(-1, pred.shape[-1])

                    adata = ad.AnnData(pred)
                    adata.obs[var] = ['predicted']*pred.shape[0]
                    adata.obs['ptb'] = [gene]*pred.shape[0]
                    adata.obs['mode'] = [mode]*pred.shape[0]
                    adata.obs['idx'] = [i] * pred.shape[0]
                    variables[var].append(adata)
        
        if not exists_nontargeting:

            adata = sc.read_h5ad('./h5ad_datafiles/k562_annotated_raw_allnontargetinginval.h5ad')

            adata = adata[adata.obs['gene'] == 'non-targeting']
            sc.pp.normalize_total(adata)
            sc.pp.log1p(adata)
            adata = adata[adata.obs['split'] == 'train'].X
            nontargeting = ad.AnnData(adata)
            nontargeting.obs['mode'] = ['nontargeting']*nontargeting.shape[0]
            nontargeting.obs['ptb'] = ['nontargeting_ref']*nontargeting.shape[0]
            nontargeting.obs['y'] = ['nontargeting_ref']*nontargeting.shape[0]
            nontargeting.obs['idx'] = [-1] * nontargeting.shape[0]

            variables['y'].append(nontargeting)
        for mode in inference_modes:
            likelihood[mode] /= total
        variables['y'] += ads_true

        for var in variables:
            if var in ['lkl', 'rates', 'dropouts', 'lib_vars', 'ptb_name']:
                continue
            variables[var] = ad.concat(variables[var])
        
        for mode in inference_modes:
            if len(rates[mode]) > 0 and len(dropouts[mode]) > 0 and len(lib_vars[mode]) > 0:
                rates[mode] = np.vstack(rates[mode])
                dropouts[mode] = np.vstack(dropouts[mode])
                lib_vars[mode] = np.vstack(lib_vars[mode])
        
        rates = np.array([x for x in rates.values()])
        dropouts = np.array([x for x in dropouts.values()])
        lib_vars = np.array([x for x in lib_vars.values()])
        return variables

    if type(shiftval) == list:
        all_vars = []
        for shift in shiftval:
            all_vars.append(compute_for_shift(shift))
        return all_vars
    
    return compute_for_shift(shiftval)

def inference(model, p, p1h, device, shiftval = None, gene = None, normed = False, hard = True):
    model.eval()

    x = model.Xs[np.random.choice(model.Xs.shape[0], p.shape[0]),:]
    x = torch.from_numpy(x).to(model.device).double()

    p_zero = torch.zeros_like(p).to(device)

    if hard:
        latents = model.compute_latents(x, p, p1h, gene = gene)
    else:
        latents = model.compute_latents(x, p, p1h, gene = gene, p_augment = p_zero,)

    if shiftval is None:
        c = latents['c']
    else:
        c = shiftval * torch.ones_like(latents['c']).to(device)

    outputs = model.reconstruct(latents['z'], latents['ptb_enc'], c)

    return {
        'y': outputs['y'],
        'p': latents['ptb_enc'],
        'p_input': p,
        'c': latents['c'],
        'z': latents['z'],
        'u': outputs['u'],
        'shift': outputs['shift'],
        'z_shift': outputs['shift'] + latents['qz_dist'].loc,
        'z_mean': latents['qz_dist'].loc,
        'u_shift': torch.matmul(outputs['shift'] + latents['qz_dist'].loc, torch.inverse(torch.eye(model.zdim).to(model.device) - model.B*model.B_mask))
    }


def eval(args):
    out_root = args['out_root']
    model_name = args['model_name']

    selection = None


    i = args['device']
    arg_save_name = args['save_name']
    device = torch.device(f"cuda:{i}" if torch.cuda.is_available() else "cpu")

    if args['ood_split'] is None:
        args['ood_split'] = -1

    variables = compute_predictions(
        model_path = f'{out_root}/{model_name}.pth',
        inference_modes = ['hard'],
        device = device,
        ptb = selection, #ptb = None will evaluate the whole set
        setting = args['graph'],
        variable = ['shift', 'z_shift'],
        ood=False,
        downsample = args['downsample'], 
        ptb_type = args['ptb_type'],
        ptb_dim = args['ptb_dim'],
        data_mode = args['data_split'],
        shiftval = args['shiftval'],
        ood_split = args['ood_split'],
        rand_graph_seed = args['rand_graph_seed'],
    )

    save_name = f'{arg_save_name}_{model_name}' # save name of adata file with generated samples

    print('SAVING THINGS:')

    for var in variables:
        if var == 'ptb_name':
            continue
        if var in ['lkl', 'rates', 'dropouts', 'lib_vars', 'theta']:
            values = variables[var]
            np.save(f'{out_root}/{save_name}_{var}.npy', values)
        else:
            adata = variables[var]
            adata.write(f'{out_root}/{save_name}_{var}.h5ad')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-m', '--out-root', default='.', type=str,
                        help='name of model weights saved directory')
    parser.add_argument('-s', '--save-name', default='inference', type=str,
                        help='name of save dir')
    parser.add_argument('-n', '--model-name', default='last_model', type=str,
                        help='name of model')
    parser.add_argument('-d', '--device', default=0, type=int,
                        help='cuda device number')
    parser.add_argument('-p', '--ptb_type', default='genept', type=str, 
                        help='Perturbation encoding type: [onehot, expression, genept]')
    parser.add_argument('--perturb-dim', default=1536, type=int,
                        help='perturbation dimension')
    parser.add_argument('-x', '--downsample', default = 1, type = float, help = 'fraction of dataset to evaluate on')
    parser.add_argument('--data-split', default = None)
    parser.add_argument('--graph', default = 'causal', type=str)
    parser.add_argument('--shiftval', default = None, type=float, required = False)
    parser.add_argument('--ood-split', default = None, type=int, required = False)
    parser.add_argument('--normed', action = 'store_true', help = 'use normed distribution')
    parser.add_argument('--hard', action = 'store_true', help = 'trained on hard inference?')
    parser.add_argument('--rand-graph-seed', default = 0, type = int, help = 'random seed for graph generation')


    seed = 20
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.set_default_dtype(torch.float64)

    args = parser.parse_args()

    arg_config = {
        'out_root': args.out_root,
        'save_name': args.save_name,
        'model_name': args.model_name, 
        'device': args.device,
        'downsample': args.downsample, 
        'ptb_type': args.ptb_type,
        'ptb_dim': args.perturb_dim,
        'data_split': args.data_split,
        'graph': args.graph,
        'shiftval': args.shiftval,
        'normed': args.normed,
        'hard': args.hard,
        'ood_split': args.ood_split,
        'rand_graph_seed': args.rand_graph_seed
    }

    eval(arg_config)