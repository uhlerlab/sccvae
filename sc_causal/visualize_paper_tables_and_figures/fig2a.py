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
import seaborn as sns


import sys
sys.path.append('../')
from tqdm import tqdm
import os
from causal.dataset import PerturbDataset
from matplotlib.colors import SymLogNorm
import json

import sys
sys.path.append('./')

# Including all metrics, 2a looks at MSE, MMD, and Frac Changed Direction
metrics = ['MSE', 'PearsonR','MMD', 'Energy Distance' , 'Frac Correct Direction', 'Frac Changed Direction']
genes = ['All genes', 'Top 50']

distribution = 'ood'
color_palette = ['#D3D3D3', '#241623', '#46A7D1']

fig, axs = plt.subplots(3, 2, figsize=(8, 12))
for i, metric in enumerate(metrics[0:1]+metrics[2:3] + metrics[5:6]):
    print(metric)
    print('All genes')

    df1 = pd.read_csv(f'tables/{distribution}_{metric}_All genes.csv')
    df1 = df1.drop(columns=['Causal', 'RandAvg', 'Conditional'])


    df1 = df1.rename(columns={'Full': 'SCCVAE', 'Nontargeting': 'Control'})
    df1 = df1.reindex(columns=['PTBs','Control', 'GEARS', 'SCCVAE'])

    boxplot1 = sns.boxplot(data=df1, ax=axs[i, 0], palette = color_palette)


    axs[i, 0].set_title(f'All essential genes', fontsize=18)
    axs[i, 0].set_ylabel(metric, fontsize=15)
    axs[i, 0].set_xticklabels(['Control', 'GEARS', 'SCCVAE'], fontsize=15)

    df2 = pd.read_csv(f'tables/{distribution}_{metric}_Top 50.csv')
    df2 = df2.drop(columns=['Causal', 'RandAvg', 'Conditional'])
    df2 = df2.rename(columns={'Full': 'SCCVAE'})
    df2 = df2.reindex(columns=['PTBs','Nontargeting', 'GEARS', 'SCCVAE'])

    print('Top 50')


    boxplot2 = sns.boxplot(data=df2, ax=axs[i, 1], palette = color_palette)

    axs[i, 1].set_title(f'Top 50 genes', fontsize=18)
    axs[i, 1].set_xticklabels(['Control', 'GEARS', 'SCCVAE'], fontsize=15)
os.makedirs('figures', exist_ok=True)
plt.savefig(f'figures/2a.png', bbox_inches = 'tight')
plt.show()
plt.clf()