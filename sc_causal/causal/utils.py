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
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import wget
import random

import sys
sys.path.append('./')

def read_adata_from_web():
    try:
        adata = sc.read_h5ad('./h5ad_datafiles/K562_essential_raw_singlecell_01.h5ad')
    except Exception as e:
        print(e)
        url = 'https://plus.figshare.com/ndownloader/files/35773219'
        datafile = wget.download(url, out = './h5ad_datafiles')
        adata = sc.read_h5ad(datafile)
    return adata

def make_multiple_splits(splits, idx):
    test_split = splits[idx]
    train_split = splits[:idx] + splits[idx+1:]
    train_split = [item for sublist in train_split for item in sublist]
    return train_split, test_split

def create_out_of_distribution_h5ad(out_name, n_splits = 5):
    adata = read_adata_from_web()
    df = pd.read_csv('./train_util_files/embeddings_logreg_scores.csv').fillna(0)
    genelist = adata.obs['gene'].value_counts().index.tolist()
    good_perturbs = []
    for gene in genelist:
        logreg = df[df['Gene'] == gene]['cv_mean'].item()
        count = df[df['Gene'] == gene]['counts'].item()
        if logreg >= 0.6 and count >= 200:
            good_perturbs.append(gene)

    random.shuffle(good_perturbs)

    all_splits = []
    for split_num in range(n_splits):
        all_splits.append([])
    
    for i, ptb in enumerate(good_perturbs):
        all_splits[i % n_splits].append(ptb)

    for split_num in range(5):
        print('doing split', split_num)
        # genes_train = good_perturbs[:int(0.9*len(good_perturbs))]
        # genes_test = good_perturbs[int(0.9*len(good_perturbs)):]
        genes_train, genes_test = make_multiple_splits(all_splits, split_num)

        assert list(set(genes_train) & set(genes_test)) == []

        train_ptbs = [adata[adata.obs['gene'] == 'non-targeting']]
        val_ptbs = []
        test_ptbs = []
        for ptb in genes_train:
            a_train, a_val = train_test_split(adata[adata.obs['gene'] == ptb], test_size = 1/8)
            train_ptbs.append(a_train)
            val_ptbs.append(a_val)

        for ptb in genes_test:
            a_test = adata[adata.obs['gene'] == ptb]
            test_ptbs.append(a_test)

        adata_train = ad.concat(train_ptbs)
        adata_val = ad.concat(val_ptbs)
        adata_test = ad.concat(test_ptbs)
        
        adata_train.obs['split'] = ['train'] * len(adata_train)
        adata_val.obs['split'] = ['val'] * len(adata_val)
        adata_test.obs['split'] = ['test'] * len(adata_test)

        adata_concat = ad.concat([adata_train, adata_val, adata_test])
        adata_concat.write_h5ad(f'./h5ad_datafiles/{out_name}_{split_num}.h5ad')

    return adata

def create_in_distribution_h5ad(out_name):
    datafile = read_adata_from_web()
    adata = sc.read_h5ad(datafile)
    df = pd.read_csv('./train_util_files/embeddings_logreg_scores.csv').fillna(0)
    genelist = adata.obs['gene'].value_counts().index.tolist()
    good_perturbs = []
    for gene in genelist:
        logreg = df[df['Gene'] == gene]['cv_mean'].item()
        count = df[df['Gene'] == gene]['counts'].item()
        if logreg >= 0.6 and count >= 200:
            good_perturbs.append(gene)

    train_ptbs = [adata[adata.obs['gene'] == 'non-targeting']]
    val_ptbs = []
    test_ptbs = []
    for ptb in good_perturbs:
        a_train, a_test = train_test_split(adata[adata.obs['gene'] == ptb], test_size = 0.2)
        a_train, a_val = train_test_split(a_train[a_train.obs['gene'] == ptb], test_size = 1/8)
        train_ptbs.append(a_train)
        val_ptbs.append(a_val)
        test_ptbs.append(a_test)

    adata_train = ad.concat(train_ptbs)
    adata_val = ad.concat(val_ptbs)
    adata_test = ad.concat(test_ptbs)
    
    adata_train.obs['split'] = ['train'] * len(adata_train)
    adata_val.obs['split'] = ['val'] * len(adata_val)
    adata_test.obs['split'] = ['test'] * len(adata_test)

    adata = ad.concat([adata_train, adata_val, adata_test])
    adata.write_h5ad(f'./h5ad_datafiles/{out_name}.h5ad')

    return adata