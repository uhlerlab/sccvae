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

import pandas as pd
import matplotlib.pyplot as plt

logreg = pd.read_csv('../train_util_files/embeddings_logreg_scores.csv')
ptbs_and_count = ((logreg['Gene'].iloc[i], logreg['counts'].iloc[i]) for i in range(len(logreg)))
ptbs_and_count_sorted = sorted(ptbs_and_count, key=lambda x: x[1], reverse=True)
ptbs = [x[0] for x in ptbs_and_count_sorted]
logregscore = [x[1] for x in ptbs_and_count_sorted]
cols = ['black' if x > 200 else 'darkgrey' for x in logregscore]

fig, ax = plt.subplots(figsize=(60, 10))

ax.bar([1.5*x for x in range(len(ptbs))], logregscore, width=1, color=cols)
plt.xticks([])
plt.xlabel('Perturbations', fontsize=40)
plt.yticks(fontsize=40)
plt.title('Number of Cells per Perturbation', fontsize=60)

plt.xlim(-1, 1.5*(len(ptbs)-1)+1)

plt.savefig('figures/6a.png', bbox_inches='tight')

logreg = logreg[logreg['counts'] > 200]

ptbs_and_mean = ((logreg['Gene'].iloc[i], logreg['cv_mean'].iloc[i]) for i in range(len(logreg)))
ptbs_and_mean_sorted = sorted(ptbs_and_mean, key=lambda x: x[1], reverse=True)
ptbs = [x[0] for x in ptbs_and_mean_sorted]
logregscore = [x[1] for x in ptbs_and_mean_sorted]
cols = ['darkred' if x > 0.6 else 'darkblue' for x in logregscore]

fig, ax = plt.subplots(figsize=(60, 10))

ax.bar([1.5*x for x in range(len(ptbs))], logregscore, width=1, color=cols)
plt.xticks([])
plt.xlabel('Perturbations', fontsize=40)
plt.yticks(fontsize=40)
plt.title('5-fold Logistic Regression Score for Perturbations with >=200 Cells', fontsize=60)

plt.xlim(-1, 1.5*(len(ptbs)-1)+1)

plt.savefig('figures/6b.png', bbox_inches='tight')