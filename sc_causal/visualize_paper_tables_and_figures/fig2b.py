import anndata as ad
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pickle
import pandas as pd
import scanpy as sc
from scipy.spatial import distance_matrix
import torch
from torch.utils.data import Dataset, DataLoader


import json
import sys
sys.path.append('../')
from tqdm import tqdm
import os
from causal.dataset import PerturbDataset
from causal.metrics import mse_per_ptb, mmd_per_ptb, pearsonr_per_ptb, frac_correct_direction_per_ptb, frac_changed_direction_per_ptb

import scvi
from matplotlib.colors import SymLogNorm

import sys
sys.path.append('./')

import argparse

def create_metrics_data(adata, ptbs):
    subset = adata[adata.obs['mode'].isin(['hard', 'nontargeting', 'ref'])]

    ret = {}

    mses = mse_per_ptb(subset)
    mmds = mmd_per_ptb(subset)
    pearsonrs = pearsonr_per_ptb(subset)
    frac_correct_directions = frac_correct_direction_per_ptb(subset)
    frac_changed_directions = frac_changed_direction_per_ptb(subset)

    ret = {}
    for ptb in [x for x in ptbs if (x != 'non-targeting' and x != 'nontargeting_ref')]:
        if ptb not in mses:
            continue
        ret[ptb] = {
            'mse': round(float(mses[ptb]), 3),
            'mmd': round(float(mmds[ptb]), 3),
            'pearsonr': round(float(pearsonrs[ptb]), 3),
            'frac_correct': round(float(frac_correct_directions[ptb]), 3),
            'frac_changed': round(float(frac_changed_directions[ptb]), 3)
        }
    return ret


gears_data = {
    "0": "../GEARS/demo/inference_anndatas/results_gears_ood_split_0_withnontargeting.h5ad",
    "1": "../GEARS/demo/inference_anndatas/results_gears_ood_split_1_withnontargeting.h5ad",
    "2": "../GEARS/demo/inference_anndatas/results_gears_ood_split_2_withnontargeting.h5ad",
    "3": "../GEARS/demo/inference_anndatas/results_gears_ood_split_3_withnontargeting.h5ad",
    "4": "../GEARS/demo/inference_anndatas//results_gears_ood_split_4_withnontargeting.h5ad"
}

out_roots = {
    "0": "outs/sccvae_0/inference_best_val_mmd_shiftselect_hard",
    "1": "outs/sccvae_1/inference_best_val_mmd_shiftselect_hard",
    "2": "outs/sccvae_2/inference_best_val_mmd_shiftselect_hard",
    "3": "outs/sccvae_3/inference_best_val_mmd_shiftselect_hard",
    "4": "outs/sccvae_4/inference_best_val_mmd_shiftselect_hard"
}

ctr = 0
fig, axs = plt.subplots(3, 3, figsize=(18, 18))

plt.suptitle('Post-perturbational expression predictions', fontsize=40)
for lbl in out_roots:
    out_root = out_roots[lbl]

    if 'h5ad' not in out_root:
        adata = sc.read_h5ad('../'+out_root + '_y.h5ad')
    else:
        adata = sc.read_h5ad(f'../{out_root}')

    print(set(adata.obs['mode']))

    gt = adata[adata.obs['y'].isin(['ground truth', 'nontargeting distribution'])].copy()
    pred = adata[adata.obs['mode'].isin(['hard', 'nontargeting distribution'])].copy()
    nontargeting = gt[gt.obs['ptb'] == 'non-targeting'].copy()

    if len(nontargeting) == 0:
        nontargeting = adata[adata.obs['ptb'] == 'nontargeting_ref'].copy()

    ctrl_X = nontargeting.X
    ptbs = [x for x in set(adata.obs['ptb']) if (x != 'non-targeting' and x != 'nontargeting_ref')]

    gears_split = sc.read_h5ad(gears_data[lbl])
    gears_split = gears_split[gears_split.obs['y'].isin(['predicted'])].copy()
    print(ptbs)
    ptbs = ['MED10', 'NUP54', 'EIF3H', 'OXA1L', 'KRR1', 'NLE1', 'XPO1', 'MED7', 'POLR2A']
    for ptb in ptbs:
        gt_ptbs = gt[gt.obs['ptb'] == ptb].copy()
        pred_ptbs  = pred[pred.obs['ptb'] == ptb].copy()
        gears_ptbs = gears_split[gears_split.obs['ptb'] == ptb].copy()


        if len(gt_ptbs) == 0 or len(pred_ptbs) == 0: 
            print(f'No data for {ptb}')
            continue

        gt_y = gt_ptbs.X
        pred_hard_y = pred_ptbs.X
        gears_y = gears_ptbs.X

        all_data = np.vstack([ctrl_X, gt_y, gears_y,  pred_hard_y])
        adata_new = sc.AnnData(all_data)

        sc.tl.pca(adata_new, svd_solver='arpack')
        sc.pp.neighbors(adata_new, n_neighbors=40, n_pcs=50)
        sc.tl.umap(adata_new) 

        label = ['NA' for _ in range(len(ctrl_X))] + ['Actual Cells' if gt_ptbs.obs['ptb'][i]==ptb else 'NA' for i in range(len(gt_ptbs))] + ['GEARS' for _ in range(len(gears_ptbs))] + ['SCCVAE' for _ in range(len(pred_ptbs))]

        adata_new.obs['label'] = label

        ax = axs[ctr//3][ctr%3]

        if ctr == 5:
            sc.pl.umap(adata_new, size=90, 
                    color=['label'],
                    legend_fontsize=30, 
                    groups=['Actual Cells', 'SCCVAE', 'GEARS'],
                    title=[ptb], 
                    palette={
                        'Actual Cells': '#E55F3F',
                        'SCCVAE': '#46A7D1',
                        'GEARS': '#241623', 
                        'NA': 'lightgrey'
                    },
                    na_in_legend=False,
                    legend_fontweight='bold',
                    wspace = 1.0,
                    ax=ax
            )
        else:
            sc.pl.umap(adata_new, size=90, 
                    color=['label'],
                    groups=['Actual Cells', 'SCCVAE', 'GEARS'],
                    title=[ptb], 
                    palette={
                        'Actual Cells': '#E55F3F',
                        'SCCVAE': '#46A7D1',
                        'GEARS': '#241623', 
                        'NA': 'lightgrey'
                    },
                    na_in_legend=False,
                    legend_loc=None,
                    wspace = 1.0,
                    ax=ax
            )


        ax.set_title(f'{ptb}', fontsize = 25)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ctr += 1
plt.savefig(f'figures/2b.png',  bbox_inches = 'tight')
