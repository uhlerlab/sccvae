import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

import scanpy as sc
import sys
sys.path.append('../')
import os
from causal.metrics import mse_per_ptb

import sys
sys.path.append('./')

import argparse

root_fname = 'outs/sccvae_3'
ptb = 'MED7'
shiftvals = [x/10 for x in range(-10, 31, 5)]

def create_metrics_data(adata, ptbs):
    subset = adata[adata.obs['mode'].isin(['hard', 'nontargeting', 'ref'])]

    ret = {}

    mses = mse_per_ptb(subset, include_error = True)

    ret = {}
    for ptb in [x for x in ptbs if (x != 'non-targeting' and x != 'nontargeting_ref')]:
        if ptb not in mses:
            continue
        ret[ptb] = {
            'mse': round(float(mses[ptb][0]), 3),
            'mse_std': round(float(mses[ptb][1]), 3),
        }
    return ret


out_roots = {}

for shift in shiftvals:
    out_roots[shift] = f'{root_fname}/cache/prediction_{shift}_hard'

all_data = []
gt = None

errors_mean = []
errors_std = []

for lbl in out_roots:
        
    out_root = out_roots[lbl]

    adata = sc.read_h5ad('../'+out_root + '.h5ad')
    metrics = create_metrics_data(adata[adata.obs['ptb'] == ptb], adata.obs['ptb'])

    errors_mean.append(metrics[ptb]['mse'])
    errors_std.append(metrics[ptb]['mse_std'])

    if gt is None:
        gt = adata[adata.obs['y'].isin(['ground truth', 'nontargeting distribution'])].copy()
        nontargeting = gt[gt.obs['ptb'] == 'non-targeting'].copy()

        if len(nontargeting) == 0:
            nontargeting = adata[adata.obs['ptb'] == 'nontargeting_ref'].copy()

        ctrl_X = nontargeting.X
        gt_ptbs = gt[gt.obs['ptb'] == ptb].copy()
        gt_y = gt_ptbs.X

        adata_new = sc.AnnData(np.vstack([ctrl_X, gt_y]))
        adata_new.obs['label'] =  ['NA' for _ in range(len(ctrl_X))] + ['Actual Cells' for _ in range(len(gt_ptbs))]
        adata_new.obs['label_shifts'] = ['NA' for _ in range(len(ctrl_X) + len(gt_y))]
        all_data.append(adata_new)
    
    pred = adata[adata.obs['mode'].isin(['hard', 'nontargeting distribution'])].copy()
    pred_ptbs  = pred[pred.obs['ptb'] == ptb].copy()

    if len(pred_ptbs) == 0:
        print(f'No data for {ptb}')
        continue

    pred_hard_y = pred_ptbs.X

    adata_new = sc.AnnData(pred_hard_y)

    label = [f'Shift = {lbl}' for _ in range(len(pred_hard_y))]
    adata_new.obs['label'] = ['NA' for _ in range(len(pred_hard_y))]

    adata_new.obs['label_shifts'] = label
    all_data.append(adata_new)

adata_new = ad.concat(all_data)

sc.tl.pca(adata_new, svd_solver='arpack')
sc.pp.neighbors(adata_new, n_neighbors=40, n_pcs=50)
sc.tl.umap(adata_new) 


idx = np.argmin(np.array(errors_mean))

fig, axs = plt.subplots(1, 3, figsize=(30, 7))

ax = axs[0]
ax.plot(shiftvals, errors_mean, label='MSE', color='black', linewidth=2)
ax.fill_between(shiftvals, [max(x - y, 0) for x, y in zip(errors_mean, errors_std)], [x + y for x, y in zip(errors_mean, errors_std)], color='black', alpha=0.1)
ax.set_xlabel('Shift', fontsize = 20)
ax.set_ylabel('MSE', fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.axvline(x = shiftvals[idx], color='red', linestyle='--', linewidth=2)
ax.set_title(f'{ptb} MSE for different shifts', fontsize = 25)
# p
color_map = {'Actual Cells': 'red',
             'NA': 'lightgrey'
             }

viridis = plt.get_cmap('viridis')
idxs = np.linspace(0, 1, len(shiftvals))

for i, shift in enumerate(shiftvals):
    color_map['Shift = ' + str(shift)] = viridis(idxs[i])

ax = axs[2]
sc.pl.umap(adata_new, size=90, 
        color=['label_shifts'],
        legend_fontsize=30, 
        groups=['Actual Cells'] + ['Shift = ' + str(x) for x in shiftvals],
        palette=color_map,
        na_in_legend = False,
        ax=ax
)
ax.set_title(f'{ptb} predictions for different shifts', fontsize = 25)
ax.set_xlabel('')
ax.set_ylabel('')
handles, labels = ax.get_legend_handles_labels()
handles, labels = plt.gca().get_legend_handles_labels()
order = [8, 7, 0, 1, 2, 3, 4, 5, 6]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, frameon=False)


ax = axs[1]
sc.pl.umap(adata_new, size=90, 
        color=['label'],
        legend_fontsize=30, 
        groups=['Actual Cells'] + ['Shift = ' + str(x) for x in shiftvals],
        palette=color_map,
        legend_loc=None,
        na_in_legend = False,
        ax=ax
)
ax.set_title(f'{ptb} ground truth', fontsize = 25)
ax.set_xlabel('')
ax.set_ylabel('')

plt.savefig(f'figures/4.png', bbox_inches='tight')




root_fname = 'outs/sccvae_2'

ptb = 'NLE1'
shiftvals = [x/10 for x in range(-10, 31, 5)]
savedir = f'{ptb}_all_shifts'

def create_metrics_data(adata, ptbs):
    subset = adata[adata.obs['mode'].isin(['hard', 'nontargeting', 'ref'])]

    ret = {}

    mses = mse_per_ptb(subset, include_error = True)

    ret = {}
    for ptb in [x for x in ptbs if (x != 'non-targeting' and x != 'nontargeting_ref')]:
        if ptb not in mses:
            continue
        ret[ptb] = {
            'mse': round(float(mses[ptb][0]), 3),
            'mse_std': round(float(mses[ptb][1]), 3),
        }
    return ret


try:
    os.mkdir(f'figures/umap/{savedir}')
except:
    print('dir already exists')


out_roots = {}

for shift in shiftvals:
    out_roots[shift] = f'{root_fname}/cache/prediction_{shift}_hard'

all_data = []
gt = None

errors_mean = []
errors_std = []

for lbl in out_roots:
        
    out_root = out_roots[lbl]

    adata = sc.read_h5ad('../'+out_root + '.h5ad')
    metrics = create_metrics_data(adata[adata.obs['ptb'] == ptb], adata.obs['ptb'])

    errors_mean.append(metrics[ptb]['mse'])
    errors_std.append(metrics[ptb]['mse_std'])

    if gt is None:
        gt = adata[adata.obs['y'].isin(['ground truth', 'nontargeting distribution'])].copy()
        nontargeting = gt[gt.obs['ptb'] == 'non-targeting'].copy()

        if len(nontargeting) == 0:
            nontargeting = adata[adata.obs['ptb'] == 'nontargeting_ref'].copy()

        ctrl_X = nontargeting.X
        gt_ptbs = gt[gt.obs['ptb'] == ptb].copy()
        gt_y = gt_ptbs.X

        adata_new = sc.AnnData(np.vstack([ctrl_X, gt_y]))
        adata_new.obs['label'] =  ['NA' for _ in range(len(ctrl_X))] + ['Actual Cells' for _ in range(len(gt_ptbs))]
        adata_new.obs['label_shifts'] = ['NA' for _ in range(len(ctrl_X) + len(gt_y))]
        all_data.append(adata_new)
    
    pred = adata[adata.obs['mode'].isin(['hard', 'nontargeting distribution'])].copy()
    pred_ptbs  = pred[pred.obs['ptb'] == ptb].copy()

    if len(pred_ptbs) == 0:
        print(f'No data for {ptb}')
        continue

    pred_hard_y = pred_ptbs.X

    adata_new = sc.AnnData(pred_hard_y)

    label = [f'Shift = {lbl}' for _ in range(len(pred_hard_y))]
    adata_new.obs['label'] = ['NA' for _ in range(len(pred_hard_y))]

    adata_new.obs['label_shifts'] = label
    all_data.append(adata_new)

adata_new = ad.concat(all_data)

sc.tl.pca(adata_new, svd_solver='arpack')
sc.pp.neighbors(adata_new, n_neighbors=40, n_pcs=50)
sc.tl.umap(adata_new) 

idx = np.argmin(np.array(errors_mean))

fig, axs = plt.subplots(1, 3, figsize=(30, 7))

ax = axs[0]
ax.plot(shiftvals, errors_mean, label='MSE', color='black', linewidth=2)
ax.fill_between(shiftvals, [max(x - y, 0) for x, y in zip(errors_mean, errors_std)], [x + y for x, y in zip(errors_mean, errors_std)], color='black', alpha=0.1)
ax.set_xlabel('Shift', fontsize = 20)
ax.set_ylabel('MSE', fontsize = 20)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.axvline(x = shiftvals[idx], color='red', linestyle='--', linewidth=2)
ax.set_title(f'{ptb} MSE for different shifts', fontsize = 25)

color_map = {'Actual Cells': 'red',
             'NA': 'lightgrey'
             }

viridis = plt.get_cmap('viridis')
idxs = np.linspace(0, 1, len(shiftvals))

for i, shift in enumerate(shiftvals):
    color_map['Shift = ' + str(shift)] = viridis(idxs[i])


ax = axs[2]
sc.pl.umap(adata_new, size=90, 
        color=['label_shifts'],
        legend_fontsize=30, 
        groups=['Actual Cells'] + ['Shift = ' + str(x) for x in shiftvals],
        palette=color_map,
        na_in_legend = False,
        ax=ax
)
ax.set_title(f'{ptb} predictions for different shifts', fontsize = 25)
ax.set_xlabel('')
ax.set_ylabel('')
handles, labels = ax.get_legend_handles_labels()
handles, labels = plt.gca().get_legend_handles_labels()
order = [8, 7, 0, 1, 2, 3, 4, 5, 6]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20, frameon=False)

ax = axs[1]
sc.pl.umap(adata_new, size=90, 
        color=['label'],
        legend_fontsize=30, 
        groups=['Actual Cells'] + ['Shift = ' + str(x) for x in shiftvals],
        palette=color_map,
        legend_loc=None,
        na_in_legend = False,
        ax=ax
)
ax.set_title(f'{ptb} ground truth', fontsize = 25)
ax.set_xlabel('')
ax.set_ylabel('')

plt.savefig(f'figures/8.png', bbox_inches='tight')
