from gene_groups_config import GENE_MODULES

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

import sys
sys.path.append('../')
import os
import json

import sys
sys.path.append('./')

selected_gene_groups = [
    'ribosomal_proteins',
    'proteasome_subunits',
    'mediator_complex_components',
    'cell_cycle_regulators',
    'dna_replication_repair',
]

INVERSE_PTB_MAPPING = {}

for k in selected_gene_groups:
    for v in GENE_MODULES[k]:
        INVERSE_PTB_MAPPING[v] = k

mean = True
var = 'u'
print(var)

meanstr = 'mean' if mean else 'all'

# for any variable except for y.
out_root = 'outs/sccvae_3/inference_best_val_mmd_shiftselect_hard'
out_train = 'outs/sccvae_3/inference_train_best_val_mmd_shiftselect_hard'
adata = sc.read_h5ad('../'+out_root + f'_{var}.h5ad')
adata_train = sc.read_h5ad('../'+out_train + f'_{var}.h5ad')

ptbs = [x for x in set(adata.obs['ptb']) if (x != 'non-targeting' and x != 'nontargeting_ref')]
ptbs_train = [x for x in set(adata_train.obs['ptb']) if (x != 'non-targeting' and x != 'nontargeting_ref')]

pred_hard = adata[adata.obs['mode'].isin(['hard', 'nontargeting distribution'])].copy()
pred_hard_train = adata_train[adata_train.obs['mode'].isin(['hard', 'nontargeting distribution'])].copy()

if not mean:
    if var not in ['p', 'p_input', 'shift']:
        pred_hard_ptbs  = pred_hard[pred_hard.obs['ptb'].isin(ptbs)].copy()
        pred_hard_train_ptbs  = pred_hard_train[pred_hard_train.obs['ptb'].isin(ptbs_train)].copy()

    else:
        pred_hard_ptbs = []
        pred_hard_train_ptbs = []
        for ptb in ptbs:
            subset = pred_hard[pred_hard.obs['ptb'] == ptb].copy()[:1]
            pred_hard_ptbs.append(subset)
        for ptb in ptbs_train:
            subset = pred_hard_train[pred_hard_train.obs['ptb'] == ptb].copy()[:1]
            pred_hard_train_ptbs.append(subset)
        pred_hard_ptbs = ad.concat(pred_hard_ptbs)
        pred_hard_train_ptbs = ad.concat(pred_hard_train_ptbs)

else:
    pred_hard_ptbs = []
    pred_hard_train_ptbs = []
    for ptb in ptbs:
        subset = np.mean(pred_hard[pred_hard.obs['ptb'] == ptb].copy().X, axis=0, keepdims=True)
        subset = ad.AnnData(subset)
        subset.obs['ptb'] = [ptb]
        pred_hard_ptbs.append(subset)
    for ptb in ptbs_train:
        subset = np.mean(pred_hard_train[pred_hard_train.obs['ptb'] == ptb].copy().X, axis=0, keepdims=True)
        subset = ad.AnnData(subset)
        subset.obs['ptb'] = [ptb]
        pred_hard_train_ptbs.append(subset)
    pred_hard_ptbs = ad.concat(pred_hard_ptbs)
    pred_hard_train_ptbs = ad.concat(pred_hard_train_ptbs)

all_ptbs = ad.concat([pred_hard_train_ptbs, pred_hard_ptbs])

sc.tl.pca(all_ptbs, svd_solver='arpack')
sc.pp.neighbors(all_ptbs, n_neighbors=40, n_pcs=50)
sc.tl.umap(all_ptbs) 

print('umapped successfully')
split = ['Test' if all_ptbs.obs['ptb'][i] in ptbs else 'Train' for i in range(len(all_ptbs))]
ptb_module = [INVERSE_PTB_MAPPING[all_ptbs.obs['ptb'][i]] if all_ptbs.obs['ptb'][i] in INVERSE_PTB_MAPPING else 'Other Genes' for i in range(len(all_ptbs))]

all_ptbs.obs['split'] = split
all_ptbs.obs['ptb_module'] = ptb_module

colors = plt.get_cmap('plasma')
idxs = np.linspace(0, 1, 2*len(selected_gene_groups)+1)

color_map = {}
for i, ptb in enumerate(selected_gene_groups):
    color_map[ptb] = colors(idxs[i*2])

color_map['Other Genes'] = 'lightgrey'

fig, ax = plt.subplots(1, 1,)

# There may be some variation in the UMAP plot due to the random initialization of the UMAP algorithm and small sample size.
# However, the clusters of perturbation modules should all be identifiable.
sc.pl.umap(all_ptbs, size=360, 
        color=['ptb_module'], 
    legend_fontsize=15, 
    groups = selected_gene_groups,
    palette = color_map,
    na_in_legend=False,
    ax=ax
)
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_title('Perturbation Modules by Average PTB Encoding', fontsize=15)
plt.savefig(f'figures/5b.png', bbox_inches = 'tight')
