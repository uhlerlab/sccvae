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
from causal.metrics import mse_per_ptb, mmd_per_ptb, pearsonr_per_ptb, frac_correct_direction_per_ptb, frac_changed_direction_per_ptb, energy_per_ptb
from causal.inference import derive_nontargeting_from_adata

import json

import sys
sys.path.append('./')



# Table columns: PTB, Nontargeting, GEARS, Causal, Conditional, Full, RandAvg

def eval_func(flag, adata, ptbs, marker_genes=False):
    if flag == 'MSE':
        return mse_per_ptb(adata, ptbs=ptbs, marker_genes=marker_genes)
    elif flag == 'MMD':
        return mmd_per_ptb(adata, ptbs=ptbs, marker_genes=marker_genes)
    elif flag == 'PearsonR':
        return pearsonr_per_ptb(adata, ptbs=ptbs, marker_genes=marker_genes)
    elif flag == 'Frac Correct Direction':
        return frac_correct_direction_per_ptb(adata, ptbs=ptbs, marker_genes=marker_genes)
    elif flag == 'Frac Changed Direction':
        return frac_changed_direction_per_ptb(adata, ptbs=ptbs, marker_genes=marker_genes)
    elif flag == 'Energy Distance':
        return energy_per_ptb(adata, ptbs=ptbs, marker_genes=marker_genes)
    else:
        raise ValueError('Invalid flag')

def eval_iid():
    iid_saved = {
        'GEARS': '../GEARS/demo/inference_anndatas/results_gears_withnontargeting.h5ad',
        'Causal': '../outs/sparse_causal_id/inference_best_val_mmd_y.h5ad',
        'Conditional': '../outs/conditional_id/inference_best_val_mmd_y.h5ad',
        'Full': '../outs/sccvae_id/inference_best_val_mmd_y.h5ad',
        'Rand1': '../outs/random_seed1_id/inference_best_val_mmd_y.h5ad',
        'Rand2': '../outs/random_seed2_id/inference_best_val_mmd_y.h5ad'
    }

    eval_func_names = ['MSE', 'MMD', 'PearsonR', 'Frac Correct Direction', 'Frac Changed Direction', 'Energy Distance']

    genes_to_eval = [
        (False, 'All genes'),
        (True, 'Top 50')
    ]

    for eval_name in eval_func_names:
        for top50_data in genes_to_eval:
            top50, top50_name = top50_data

            print(f'Running {eval_name} for {top50_name}')

            df_iid = {
                'PTBs': [],
                'Nontargeting': [],
                'GEARS': [],
                'Causal': [],
                'Conditional': [],
                'Full': [],
                'RandAvg': [],
            }

            gears = ad.read(iid_saved['GEARS'])
            excl = {'nontargeting', 'nontargeting_ref', 'INTS2', 'POLR2B', 'SMN2', 'DDX47', 'TREX2', 'PHB', 'BGLAP', 'ANAPC2', 'ZNF687'}
            ptbs = [x for x in list(set(gears.obs['ptb'])) if not (x in excl)]
            df_iid['PTBs'] = ptbs

            gears_errs = eval_func(eval_name, gears, ptbs=ptbs, marker_genes=top50)
            df_iid['GEARS'] = [gears_errs[x] for x in ptbs]

            nontargeting = derive_nontargeting_from_adata(gears[gears.obs['y'].isin(['ground truth', 'nontargeting_ref'])])
            nontargeting_errs = eval_func(eval_name, nontargeting, ptbs=ptbs, marker_genes=top50)
            df_iid['Nontargeting'] = [nontargeting_errs[x] for x in ptbs]

            causal = sc.read_h5ad(iid_saved['Causal'])
            causal_errs = eval_func(eval_name, causal, ptbs=ptbs, marker_genes=top50)
            df_iid['Causal'] = [causal_errs[x] for x in ptbs]

            conditional = sc.read_h5ad(iid_saved['Conditional'])
            conditional_errs = eval_func(eval_name, conditional, ptbs=ptbs, marker_genes=top50)
            df_iid['Conditional'] = [conditional_errs[x] for x in ptbs]

            full = sc.read_h5ad(iid_saved['Full'])
            full_errs = eval_func(eval_name, full, ptbs=ptbs, marker_genes=top50)
            df_iid['Full'] = [full_errs[x] for x in ptbs]

            rand1 = sc.read_h5ad(iid_saved['Rand1'])
            rand1_errs = eval_func(eval_name, rand1, ptbs=ptbs, marker_genes=top50)

            rand2 = sc.read_h5ad(iid_saved['Rand2'])
            rand2_errs = eval_func(eval_name, rand2, ptbs=ptbs, marker_genes=top50)

            df_iid['RandAvg'] = [(rand1_errs[x] + rand2_errs[x]) / 2 for x in ptbs]

            df_iid = pd.DataFrame(df_iid)
            df_iid.to_csv(f'./tables/iid_{eval_name}_{top50_name}.csv', index=False)

def eval_ood():
    ood_saved_split_0 = {
        'GEARS': '../GEARS/demo/inference_anndatas/results_gears_ood_split_0_withnontargeting.h5ad',
        'Causal': '../outs/sparse_causal_0/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Conditional': '../outs/conditional_0/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Full': '../outs/sccvae_0/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand1': '../outs/random_seed1_0/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand2': '../outs/random_seed2_0/inference_best_val_mmd_shiftselect_hard_y.h5ad'
    }

    ood_saved_split_1 = {
        'GEARS': '../GEARS/demo/inference_anndatas/results_gears_ood_split_1_withnontargeting.h5ad',
        'Causal': '../outs/sparse_causal_1/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Conditional': '../outs/conditional_1/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Full': '../outs/sccvae_1/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand1': '../outs/random_seed1_1/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand2': '../outs/random_seed2_1/inference_best_val_mmd_shiftselect_hard_y.h5ad'
    }

    ood_saved_split_2 = {
        'GEARS': '../GEARS/demo/inference_anndatas/results_gears_ood_split_2_withnontargeting.h5ad',
        'Causal': '../outs/sparse_causal_2/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Conditional': '../outs/conditional_2/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Full': '../outs/sccvae_2/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand1': '../outs/random_seed1_2/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand2': '../outs/random_seed2_2/inference_best_val_mmd_shiftselect_hard_y.h5ad'
    }

    ood_saved_split_3 = {
        'GEARS': '../GEARS/demo/inference_anndatas/results_gears_ood_split_3_withnontargeting.h5ad',
        'Causal': '../outs/sparse_causal_3/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Conditional': '../outs/conditional_3/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Full': '../outs/sccvae_3/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand1': '../outs/random_seed1_3/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand2': '../outs/random_seed2_3/inference_best_val_mmd_shiftselect_hard_y.h5ad'
    }

    ood_saved_split_4 = {
        'GEARS': '../GEARS/demo/inference_anndatas/results_gears_ood_split_4_withnontargeting.h5ad',
        'Causal': '../outs/sparse_causal_4/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Conditional': '../outs/conditional_4/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Full': '../outs/sccvae_4/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand1': '../outs/random_seed1_4/inference_best_val_mmd_shiftselect_hard_y.h5ad',
        'Rand2': '../outs/random_seed2_4/inference_best_val_mmd_shiftselect_hard_y.h5ad'
    }

    ood_saved = [
        ood_saved_split_0,
        ood_saved_split_1,
        ood_saved_split_2,
        ood_saved_split_3,
        ood_saved_split_4
    ]

    eval_funcs_and_names = ['MSE', 'MMD', 'PearsonR', 'Frac Correct Direction', 'Frac Changed Direction', 'Energy Distance']

    genes_to_eval = [
        (False, 'All genes'),
        (True, 'Top 50')
    ]

    for eval_name in eval_funcs_and_names:
        for top50_data in genes_to_eval:
            top50, top50_name = top50_data

            print(f'Running {eval_name} for {top50_name}')

            dfs = []

            for i, ood_saved_split_i in enumerate(ood_saved):
                print(f'Running split {i}')
                df_split = {
                    'PTBs': [],
                    'Nontargeting': [],
                    'GEARS': [],
                    'Causal': [],
                    'Conditional': [],
                    'Full': [],
                    'RandAvg': [],
                }

                gears = ad.read(ood_saved_split_i['GEARS'])
                excl = {'nontargeting', 'nontargeting_ref', 'INTS2', 'POLR2B', 'SMN2', 'DDX47', 'TREX2', 'PHB', 'BGLAP', 'ANAPC2', 'ZNF687'}
                ptbs = [x for x in list(set(gears.obs['ptb'])) if not (x in excl)]
                df_split['PTBs'] = ptbs

                gears_errs = eval_func(eval_name, gears, ptbs=ptbs, marker_genes=top50)
                df_split['GEARS'] = [gears_errs[x] for x in ptbs]

                nontargeting = derive_nontargeting_from_adata(gears[gears.obs['y'].isin(['ground truth', 'nontargeting_ref'])])
                nontargeting_errs = eval_func(eval_name, nontargeting, ptbs=ptbs, marker_genes=top50)
                df_split['Nontargeting'] = [nontargeting_errs[x] for x in ptbs]

                causal = sc.read_h5ad(ood_saved_split_i['Causal'])
                causal_errs = eval_func(eval_name, causal, ptbs=ptbs, marker_genes=top50)
                df_split['Causal'] = [causal_errs[x] for x in ptbs]

                conditional = sc.read_h5ad(ood_saved_split_i['Conditional'])
                conditional_errs = eval_func(eval_name, conditional, ptbs=ptbs, marker_genes=top50)
                df_split['Conditional'] = [conditional_errs[x] for x in ptbs]

                full = sc.read_h5ad(ood_saved_split_i['Full'])
                full_errs = eval_func(eval_name, full, ptbs=ptbs, marker_genes=top50)
                df_split['Full'] = [full_errs[x] for x in ptbs]

                rand1 = sc.read_h5ad(ood_saved_split_i['Rand1'])
                rand1_errs = eval_func(eval_name, rand1, ptbs=ptbs, marker_genes=top50)

                rand2 = sc.read_h5ad(ood_saved_split_i['Rand2'])
                rand2_errs = eval_func(eval_name, rand2, ptbs=ptbs, marker_genes=top50)

                df_split['RandAvg'] = [(rand1_errs[x] + rand2_errs[x]) / 2 for x in ptbs]

                df_split = pd.DataFrame(df_split)

                dfs.append(df_split)
            dfs = pd.concat(dfs)
            dfs.to_csv(f'./tables/ood_{eval_name}_{top50_name}.csv', index=False)

if __name__ == '__main__':
    if not os.path.isdir('tables'):
        os.mkdir('tables')
    eval_iid()
    eval_ood()