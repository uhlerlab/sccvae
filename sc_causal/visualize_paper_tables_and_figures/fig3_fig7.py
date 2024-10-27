import numpy as np
import torch
from tqdm import tqdm
import os
import wandb

from torch.utils.data import DataLoader

import sys
sys.path.append('../')

from causal.model import CausalVAE_Gaussian as CausalVAE
from causal.run import get_data
from causal.dataset import SCDATA_sampler
from causal.inference import compute_predictions
from causal.metrics import mse_per_ptb, mmd_per_ptb
import scanpy as sc
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine, euclidean, jensenshannon, mahalanobis, minkowski, seuclidean, sqeuclidean

def generate_nontargeting_outputs(savedir, dataset, net, ptb_dim, ptb_type, inference_mode = 'easy', var = 'p'):

    ptbs = []
    ptb_ids = []
    ptb_encodes = []

    loader_test =  DataLoader(dataset, batch_sampler=SCDATA_sampler(dataset, 32), num_workers=0)
    # ptbs_test = [x for x in list(set(dataset.data['ptb'])) if x != 'non-targeting']
    ptbs_test = ['non-targeting']

    # evaluate on TINY TEST dataset
    prediction_vars_test = compute_predictions(
        model_path = os.path.join(savedir, 'best_val_mmd_shiftselect_hard.pth'),
        inference_modes = [inference_mode],
        device = net.device,
        ptb = ptbs_test,
        ptb_dim = ptb_dim, 
        variable=var, 
        setting = 'causal', 
        ptb_type = ptb_type,
        data = (dataset, loader_test),
        parent_dir = True
    )
    p_encodes_test = np.vstack([prediction_vars_test[var][i*32].X for i in range(len(prediction_vars_test[var])//32)]) # only do 1 per batch
    print('p test shape', p_encodes_test.shape) # m x 512

    print(p_encodes_test.sum(axis=1))

    # reset ptbs_test so it corresponds with the output ptbs
    ptbs_test = [prediction_vars_test['ptb_name'][i] for i in range(len(prediction_vars_test['ptb_name']))]
    ptb_ids_test = [np.argmax(dataset.get_ptb_per_gene(x)[1]) for x in ptbs_test] # get corresponding indices in s

    ptbs.extend(ptbs_test)
    ptb_ids.extend(ptb_ids_test)
    ptb_encodes.extend(p_encodes_test)

    return {
        'ptb_names': ptbs_test,
        'ptb_ids': ptb_ids_test,
        'p_encodes': p_encodes_test,
    }

def generate_ptb_values(savedir, datasets, net, ptb_dim, ptb_type, inference_mode = 'easy', var = 'p'):

    ptbs = []
    ptb_ids = []
    ptb_encodes = []
    modes = []

    for lbl in datasets:
        print(lbl)
        dataset = datasets[lbl]
        loader_test =  DataLoader(dataset, batch_sampler=SCDATA_sampler(dataset, 32), num_workers=0)
        ptbs_test = [x for x in list(set(dataset.data['ptb'])) if x != 'non-targeting']
        print(len(ptbs_test))

        # evaluate on TINY TEST dataset
        prediction_vars_test = compute_predictions(
            model_path = os.path.join(savedir, 'best_val_mmd_shiftselect_hard.pth'),
            inference_modes = [inference_mode],
            device = net.device,
            ptb = ptbs_test,
            ptb_dim = ptb_dim, 
            variable=var, 
            setting = 'causal', 
            ptb_type = ptb_type,
            data = (dataset, loader_test),
            parent_dir = True
        )
        p_encodes_test = np.vstack([prediction_vars_test[var][i*32].X for i in range(len(prediction_vars_test[var])//32)]) # only do 1 per batch
        print('p test shape', p_encodes_test.shape) # m x 512
        print(p_encodes_test.sum(axis=1))

        # reset ptbs_test so it corresponds with the output ptbs
        ptbs_test = [prediction_vars_test['ptb_name'][i] for i in range(len(prediction_vars_test['ptb_name']))]
        ptb_ids_test = [np.argmax(dataset.get_ptb_per_gene(x)[1]) for x in ptbs_test] # get corresponding indices in s

        print(len(ptbs_test))
        print(len(ptb_ids_test))

        ptbs.extend(ptbs_test)
        ptb_ids.extend(ptb_ids_test)
        ptb_encodes.extend(p_encodes_test)
        modes.extend([lbl for _ in range(len(ptbs_test))])

    ptb_ids = np.array(ptb_ids)
    ptb_encodes = np.array(ptb_encodes)
    
    return {
        'ptb_names': ptbs,
        'ptb_ids': ptb_ids,
        'p_encodes': ptb_encodes,
        'modes': modes,
    }


def get_dists_from_node(graph, startnode):
    dists = np.zeros(graph.shape[0])
    dists.fill(graph.shape[0])
    dists[startnode] = 0

    queue = [startnode]
    seen = set([startnode])

    while len(queue) > 0:
        node = queue.pop(0)
        for i in range(graph.shape[0]):
            if i in seen:
                continue
            if graph[node, i] == 1:
                dists[i] = min(dists[i], dists[node] + 1)
                queue.append(i)
                seen.add(i)
    return dists

def generate_closeness_matrix_graph(graph_fname):
    graph = np.load(graph_fname)
    print('graph shape', graph.shape)

    dists = []
    for i in range(graph.shape[0]):
        if (i+1) % 100 == 0:
            print('getting dists from node', i+1-100,'to', i+1)
        dists.append(get_dists_from_node(graph, i))
    dists = np.vstack(dists)

    dists = np.triu(dists)
    return dists


savedirs = [
    (0, '../outs/sccvae_0', 'pca', 50, 'full', None),
    (1, '../outs/sccvae_1', 'pca', 50, 'full', None),
    (2, '../outs/sccvae_2', 'pca', 50, 'full', None),
    (3, '../outs/sccvae_3', 'pca', 50, 'full', None),
    (4, '../outs/sccvae_4', 'pca', 50, 'full', None),
]

datas_and_models = []

for batch in savedirs:
    i, savedir, ptb_type, ptb_dim, graph_mode, seed = batch

    dataset, _, _ = get_data(
        batch_size=32, 
        nontargeting=False, 
        ptb_type=ptb_type, 
        ptb_dim=ptb_dim, 
        ood=True if i is not None else False, 
        finetune=False, 
        marker=False, 
        modes = ['train', 'test'],
        split_num=i,
        tiny = True,
        parent_dir = True # Running this from sc_causal
        )

    net = CausalVAE(
        ptb_dim=ptb_dim,
        exp_dim = 8563,
        z_dim=512,
        enc_hidden = [1024, 1024], 
        dec_hidden = [1024, 1024], 
        B_filename='../train_util_files/B_512_upper_triangular.npy',
        device = 'cuda:0',
        mode = graph_mode,
        ptb_encode = ptb_type,
        rand_graph_seed=seed,
        parent_dir = True
    )
    model_path = f'{savedir}/best_val_mmd_shiftselect_hard.pth'

    state_dict = torch.load(model_path)['model_state']

    # state_dict['s'] = 0
    net.load_state_dict(state_dict)

    datas_and_models.append((dataset, net))


for b1, b2 in zip(savedirs, datas_and_models):
    split, savedir, ptb_type, ptb_dim, _, _ = b1
    print(savedir, ptb_type, ptb_dim)
    dataset, net = b2
    var = 'u_shift'

    perturbations = generate_ptb_values(savedir, dataset, net, ptb_dim, ptb_type, inference_mode = 'hard', var = var)
    assert len(perturbations['ptb_names']) == len(perturbations['ptb_ids']), print(len(perturbations['ptb_names']), len(perturbations['ptb_ids']))
    assert perturbations['p_encodes'].shape[0] == len(perturbations['ptb_names']), print(perturbations['p_encodes'].shape[0], len(perturbations['ptb_names']))
    n_ptbs = len(perturbations['ptb_names'])

    print(n_ptbs)

    nontargeting = generate_nontargeting_outputs(savedir, dataset['train'], net, ptb_dim, ptb_type, inference_mode = 'hard', var = var)

    mmds_and_p_enc_df = {
        'ptb': [],
        'mmd': [],
        'braycurtis': [],
        'canberra': [],
        'chebyshev': [],
        'cityblock': [],
        'correlation': [],
        'cosine': [],
        'euclidean': [],
        'jensenshannon': [],
        'minkowski_p1.5': [],
        'sqeuclidean': [],
        'mode': []
    }


    yvals = dataset['train'].data['y'].detach().cpu().numpy()
    ptb_idx = dataset['train'].data['ptb']

    yvals_test = dataset['test'].data['y'].detach().cpu().numpy()
    ptb_idx_test = dataset['test'].data['ptb']

    adata_nontargeting = sc.AnnData(yvals[ptb_idx == 'non-targeting'])
    adata_ptb = sc.AnnData(yvals[ptb_idx != 'non-targeting'])
    adata_ptb_test = sc.AnnData(yvals_test[ptb_idx_test != 'non-targeting'])

    adata = sc.concat([adata_nontargeting, adata_ptb, adata_ptb_test])
    adata.obs['y'] = ['ground truth' for _ in range(adata_nontargeting.shape[0])] + ['predicted' for _ in range(adata_ptb.shape[0])] + ['predicted' for _ in range(adata_ptb_test.shape[0])]
    adata.obs['ptb'] = ['nontargeting_ref' for _ in range(adata_nontargeting.shape[0])] + [ptb_idx[_] for _ in range(adata_ptb.shape[0])] + [ptb_idx_test[_] for _ in range(adata_ptb_test.shape[0])]

    # print(adata)

    mmd = mmd_per_ptb(adata)

    # print('mmd', mmd)

    for i in range(n_ptbs):
        ptb = perturbations['ptb_names'][i]

        mmds_and_p_enc_df['ptb'].append(perturbations['ptb_names'][i])
        mmds_and_p_enc_df['mmd'].append(mmd[ptb])
        mmds_and_p_enc_df['braycurtis'].append(braycurtis(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['canberra'].append(canberra(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['chebyshev'].append(chebyshev(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['cityblock'].append(cityblock(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['correlation'].append(correlation(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['cosine'].append(cosine(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['euclidean'].append(euclidean(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['jensenshannon'].append(jensenshannon(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['minkowski_p1.5'].append(minkowski(perturbations['p_encodes'][i], nontargeting['p_encodes'][0], p=1.5))
        mmds_and_p_enc_df['sqeuclidean'].append(sqeuclidean(perturbations['p_encodes'][i], nontargeting['p_encodes'][0]))
        mmds_and_p_enc_df['mode'].append(perturbations['modes'][i])

    df = pd.DataFrame(mmds_and_p_enc_df)
    print(f'Saving to causal_graph_nodes_{split}_mmd_{var}_shiftselected.csv')
    os.makedirs('scatter plot tables', exist_ok = True)
    df.to_csv(f'scatter plot tables/causal_graph_nodes_retrain_{split}_mmd_{var}_shiftselected.csv')

from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

metric = 'euclidean'

fig, axs = plt.subplots(1,5, figsize=(23, 4))
for split in range(5):
    mmd_p_enc = pd.read_csv(f'scatter plot tables/causal_graph_nodes_retrain_{split}_mmd_u_shift_shiftselected.csv')

    mmds = list(mmd_p_enc['mmd'])
    p_encs = list(mmd_p_enc[metric])

    print(split, pearsonr(mmds, p_encs)[0])
    ax = axs[split]

    ax.scatter(mmds, p_encs, color = 'black')
    ax.plot(np.unique(mmds), np.poly1d(np.polyfit(mmds, p_encs, 1))(np.unique(mmds)), '--', color = 'red')
    ax.text(np.min(mmds) + (np.max(mmds) - np.min(mmds))*0.55, 
            np.min(p_encs) + (np.max(p_encs) - np.min(p_encs))*0.55,
            'r = ' + str(round(pearsonr(mmds, p_encs)[0], 3)), fontsize=20, color = 'red')
    ax.set_xlabel(f'MMD', fontsize=20)
    if split == 0:
        ax.set_ylabel('L2 Distance', fontsize=20)
    ax.set_title(f'Split = {split}', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=13)
plt.savefig('figures/3.png', bbox_inches='tight')
plt.show()




fig, axs = plt.subplots(1,5, figsize=(23, 4))
for split in range(5):
    mmd_p_enc = pd.read_csv(f'scatter plot tables/causal_graph_nodes_retrain_{split}_mmd_u_shift_shiftselected.csv')

    mmd_p_enc = mmd_p_enc[mmd_p_enc['mode'] == 'train']

    mmds = list(mmd_p_enc['mmd'])
    p_encs = list(mmd_p_enc[metric])

    print(split, pearsonr(mmds, p_encs)[0])
    ax = axs[split]

    ax.scatter(mmds, p_encs, color = 'black')
    ax.plot(np.unique(mmds), np.poly1d(np.polyfit(mmds, p_encs, 1))(np.unique(mmds)), '--', color = 'red')
    ax.text(np.min(mmds) + (np.max(mmds) - np.min(mmds))*0.55, 
            np.min(p_encs) + (np.max(p_encs) - np.min(p_encs))*0.55,
            'r = ' + str(round(pearsonr(mmds, p_encs)[0], 3)), fontsize=20, color = 'red')
    ax.set_xlabel(f'MMD', fontsize=20)
    if split == 0:
        ax.set_ylabel('L2 Distance', fontsize=20)
    ax.set_title(f'Split = {split}', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=13)
plt.savefig('figures/7a.png', bbox_inches='tight')
plt.show()


fig, axs = plt.subplots(1,5, figsize=(23, 4))
for split in range(5):
    mmd_p_enc = pd.read_csv(f'scatter plot tables/causal_graph_nodes_retrain_{split}_mmd_u_shift_shiftselected.csv')

    mmd_p_enc = mmd_p_enc[mmd_p_enc['mode'] == 'test']

    mmds = list(mmd_p_enc['mmd'])
    p_encs = list(mmd_p_enc[metric])

    print(split, pearsonr(mmds, p_encs)[0])
    ax = axs[split]

    ax.scatter(mmds, p_encs, color = 'black')
    ax.plot(np.unique(mmds), np.poly1d(np.polyfit(mmds, p_encs, 1))(np.unique(mmds)), '--', color = 'red')
    ax.text(np.min(mmds) + (np.max(mmds) - np.min(mmds))*0.55, 
            np.min(p_encs) + (np.max(p_encs) - np.min(p_encs))*0.55,
            'r = ' + str(round(pearsonr(mmds, p_encs)[0], 3)), fontsize=20, color = 'red')
    ax.set_xlabel(f'MMD', fontsize=20)
    if split == 0:
        ax.set_ylabel('L2 Distance', fontsize=20)
    ax.set_title(f'Split = {split}', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=13)
plt.savefig('figures/7b.png', bbox_inches='tight')
plt.show()