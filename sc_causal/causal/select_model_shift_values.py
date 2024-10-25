import numpy as np
import torch
import os

from torch.utils.data import DataLoader

import sys
sys.path.append('./')

from causal.model import CausalVAE_Gaussian as CausalVAE
from causal.run import get_data
from causal.dataset import SCDATA_sampler
from causal.inference import compute_predictions
from causal.metrics import mse_per_ptb
import scanpy as sc



def model_select_inference(savedir, dataset, net, ptb_dim, ptb_type, inference_mode = 'hard', tiny = False):
    shiftvals = [0, 0.1]#[x/10 for x in range(-10, 31, 1)] # Found to be the range of learned shift vals, approx

    # first, get the set of perturbs in the dataset
    ptbs = [x for x in list(set(dataset.data['ptb'])) if x != 'non-targeting']
    ptb_ids = [np.argmax(dataset.get_ptb_per_gene(x)[1]) for x in ptbs]# get corresponding indices in s

    loader =  DataLoader(dataset, batch_sampler=SCDATA_sampler(dataset, 32), num_workers=0)

    prediction_vars = compute_predictions(
        model_path = os.path.join(savedir, 'best_val_mmd.pth'),
        inference_modes = [inference_mode],
        device = net.device,
        ptb = ptbs,
        shiftval = shiftvals,
        ptb_dim = ptb_dim, 
        variable='y', 
        # setting = 'causal', 
        ptb_type = ptb_type,
        data = (dataset, loader),
        net = net
    )
    predictions = [elem['y'] for elem in prediction_vars]

    if not os.path.exists(os.path.join(savedir, 'cache')):
        os.mkdir(os.path.join(savedir, 'cache'))
    
    for shift in shiftvals:
        savename = f'prediction_{shift}_{inference_mode}.h5ad' if not tiny else f'prediction_{shift}_{inference_mode}_tiny.h5ad'
        predictions[shiftvals.index(shift)].write(os.path.join(savedir, 'cache', savename))
    
    print('saving done!')


def model_select_errors(savedir, dataset, net, inference_mode):
    shiftvals = [0, 0.1]# [x/10 for x in range(-10, 31, 1)]

    ptbs = sorted([x for x in list(set(dataset.data['ptb'])) if x != 'non-targeting'])
    print(ptbs)
    ptb_ids = [np.argmax(dataset.get_ptb_per_gene(x)[1]) for x in ptbs] # get corresponding indices in s

    print(ptbs)

    predictions = []
    for shift in shiftvals:
        adata = sc.read_h5ad(os.path.join(savedir, 'cache', f'prediction_{shift}_{inference_mode}.h5ad'))
        predictions.append(adata)

    all_errs = []

    for prediction in predictions:
        # prediction = normalize_all_1(prediction)
        err_dict = mse_per_ptb(prediction, ptbs)
        errs = [err_dict[x] for x in err_dict]
        all_errs.append(errs)
    all_errs = np.array(all_errs) # ptbs x shift candidates
    # all_errs = np.nan_to_num(all_errs, nan=10) # some values get messed up for large shift vals and are nan, so replace with large mse value
    print(all_errs.shape)
    np.save(os.path.join(savedir, f'errs_{inference_mode}_redo_mse.npy'), all_errs)
    shiftvals = np.array(shiftvals)
    best_shifts = shiftvals[np.argmin(all_errs, axis=0)]
    for ptb_id, best_shift in zip(ptb_ids, best_shifts):
        net.s.data[ptb_id] = best_shift

    net.save(os.path.join(savedir, f'best_val_mmd_shiftselect_{inference_mode}.pth'))
    print('model saved!!')


"""
ood_split_num: [0, 1, 2, 3, 4], if OOD split. Otherwise, None
savedir: path to the saved model DIRECTORY.
graph_mode: 'full', 'causal', 'conditional', or 'random'
seed: random graph seed (only if graph_mode is random, otherwise None)
"""

SAVEDIRS = [
    #(ood_split_num, savedir, graph_mode, seed)

]

datas_and_models = []

for batch in SAVEDIRS:
    i, savedir, graph_mode, seed = batch

    dataset, _, _ = get_data(
        batch_size=32, 
        nontargeting=False, 
        ptb_type='pca', 
        ptb_dim=50, 
        ood=True if i is not None else False, 
        finetune=False, 
        marker=False, 
        modes = ['test'],
        split_num=i,
        # tiny = True
        )

    net = CausalVAE(
        ptb_dim=50,
        exp_dim = 8563,
        z_dim=512,
        enc_hidden = [1024, 1024], 
        dec_hidden = [1024, 1024], 
        B_filename='./train_util_files/B_512_upper_triangular.npy',
        device = 'cuda:0',
        mode = graph_mode,
        ptb_encode = 'pca',
        rand_graph_seed=seed
    )
    model_path = f'{savedir}/best_val_mmd.pth'

    state_dict = torch.load(model_path)['model_state']

    # state_dict['s'] = 0
    net.load_state_dict(state_dict)

    datas_and_models.append((dataset, net))

    for inference_mode in ['hard']:
        print(inference_mode, savedir, 'pca', 50)
        model_select_inference(savedir, dataset['test'], net, 50, 'pca', inference_mode)

for b1, b2 in zip(SAVEDIRS, datas_and_models):
    i, savedir, _, _ = b1
    dataset, net = b2

    for inference_mode in ['hard']:
        model_select_errors(savedir, dataset['test'], net, inference_mode)