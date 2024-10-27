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
    'mediator_complex_components',
    'cell_cycle_regulators',
    'ribosomal_proteins_60S',
    'ribosomal_proteins_40S',
    'splicing_factors',
    'splicing_rna_binding',
    'chaperones_heat_shock_proteins',
    'ribosome_biogenesis',
    'proteasome_subunits',
    'chromatin_dna_interaction',
    'dna_replication_repair',
    'chromatin_remodeling',
]

INVERSE_PTB_MAPPING = {}

for k in selected_gene_groups:
    for v in GENE_MODULES[k]:
        INVERSE_PTB_MAPPING[v] = k

mean = False
var = 'u_shift'
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

pred_hard_ptbs  = pred_hard[pred_hard.obs['ptb'].isin(ptbs)].copy()
pred_hard_train_ptbs  = pred_hard_train[pred_hard_train.obs['ptb'].isin(ptbs_train)].copy()

all_ptbs = ad.concat([pred_hard_train_ptbs, pred_hard_ptbs])

sc.tl.pca(all_ptbs, svd_solver='arpack')
sc.pp.neighbors(all_ptbs, n_neighbors=40, n_pcs=50)
sc.tl.umap(all_ptbs) 

print('umapped successfully')
split = ['Test' if all_ptbs.obs['ptb'][i] in ptbs else 'Train' for i in range(len(all_ptbs))]

all_ptbs.obs['split'] = split



fig, axs = plt.subplots(3, 4, figsize = (30, 20))
for i, gene_group in enumerate(selected_gene_groups):
    relevant_ptbs = GENE_MODULES[gene_group]
    all_ptbs.obs['ptb_module'] = [gene_group if x in relevant_ptbs else split[i] for i, x in enumerate(all_ptbs.obs['ptb'])]

    color_map = {
        gene_group: 'red',
        'Test': 'lightsteelblue',
        'Train': 'lightgrey'
    }

    ax = axs[i//4, i%4]
        
    # Titles are replaced.
    sc.pl.umap(all_ptbs, size=90, 
            color=['ptb_module'], 
        legend_fontsize=15, 
        groups = selected_gene_groups + ['Train', 'Test'],
        palette = color_map,
        na_in_legend=False,
        legend_loc = None,
        title = gene_group,
        ax=ax
    )
    ax.set_xlabel('')
    ax.set_ylabel('')
plt.savefig(f'figures/9.png', bbox_inches = 'tight')
