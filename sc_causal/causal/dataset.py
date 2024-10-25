import numpy as np
import pandas as pd
import torch
import csv
import anndata as ad

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import random
import pickle
import scanpy as sc
import os
from sklearn.decomposition import PCA


import sys
sys.path.append('../')

top_50_marker = [8562, 2858, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2859, 
                 2875, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2843, 2842, 
                 2841, 2840, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824]

class PerturbDataset(Dataset):
    def __init__(self, perturbfile, gene_id_file, mode, ptb_type='onehot', ptb_dim = 512, include_nontargeting = None, marker_genes = False):
        '''
        perturbfile: h5ad file containing perturbation data
        gene_id_file: csv file containing gene id
        mode: 'train' or 'val' or 'test'
        ptb_type: 'onehot' or 'expression' (add more later)
        '''

        if include_nontargeting is None:
            if mode == 'train':
                include_nontargeting = True
            else:
                include_nontargeting = False
        
        assert not (include_nontargeting == False and mode == 'train'), 'Must have nontargeting to train'

        torch.set_default_dtype(torch.float64)
        super(Dataset, self).__init__()

        self.ptb_type = ptb_type
        self.ptb_dim = ptb_dim

        adata = sc.read_h5ad(perturbfile)

        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)

        self.mode = mode

        if mode != 'non-targeting':
            nontargeting = adata[adata.obs['gene'] == 'non-targeting']
            adata = adata[adata.obs['split'] == mode]
            if include_nontargeting and mode != 'train':
                adata = ad.concat([adata, nontargeting])
            
        else:
            adata = adata[adata.obs['gene'] == 'non-targeting']

        genelist = adata.obs['gene'].value_counts().index.tolist()

        self.gene_exps = {}
        self.gene_exps_1h = {}

        idxlist = pd.read_csv(gene_id_file)

        full_adata = sc.read_h5ad('./h5ad_datafiles/k562_annotated_raw.h5ad')
        sc.pp.normalize_total(full_adata)
        sc.pp.log1p(full_adata)
        nontargeting = full_adata[full_adata.obs['gene'] == 'non-targeting']
        expressions = nontargeting.X
        del full_adata
        del nontargeting
        self.nontargeting = torch.from_numpy(expressions)

        pca = PCA(n_components=ptb_dim)
        pca.fit(expressions.T)
        self.pca_embeddings_50 = pca.transform(expressions.T)

        ptb_ids = []
        ys = []

        warning = []
        inverse_frequencies = []

        self.pca_embeddings = {}

        for gene in genelist:
            if gene == 'non-targeting':
                self.gene_exps_1h[gene] = np.zeros((1, 513))
                self.gene_exps_1h[gene][0, 512] = 1
                self.gene_exps[gene] = np.zeros((1, 10691))
            else:
                if len(idxlist[idxlist['gene'] == gene]) == 0: # filter genes not being considered
                    warning.append(gene)
                    continue

                self.gene_exps_1h[gene] = np.zeros((1, 513))
                self.gene_exps_1h[gene][0, idxlist[idxlist['gene'] == gene]['idx'].item()] = 1
                idx = idxlist[idxlist['gene'] == gene]['idx'].item()
                self.gene_exps[gene] = expressions[:, idx:idx+1].T
                self.pca_embeddings[gene] = self.pca_embeddings_50[idx:idx+1, :]
            
            y = adata.X[adata.obs['gene'] == gene] # n by 8563
            ptb_ids.extend([gene]*y.shape[0])
            ys.append(y)

            inverse_frequencies.extend([1/y.shape[0]] * y.shape[0])
        
        warning = ', '.join(warning)
        print(f'Warning: genes {warning} not in idxlist')
        ys = np.concatenate(ys, axis=0)
        ptb_ids = np.array(ptb_ids)

        ys = torch.from_numpy(ys)

        self.n = ys.shape[0]

        self.data = {
            'y': ys.double(),
            'ptb': ptb_ids
        }

        self.sample_freqs = torch.Tensor(inverse_frequencies)

        self.pca_embeddings['non-targeting'] = np.zeros((1, self.ptb_dim))

    def __getitem__(self, item):
        gene = self.data['ptb'][item]
        p, p1h = self.get_ptb_per_gene(gene)

        y = self.data['y'][item]
        return y, p.double(), p1h.double(), gene
        
    def __len__(self):
        return self.n
    
    def get_geneslist(self):
        return list(self.gene_exps.keys())

    def get_ptb_per_gene(self, gene):
        p1h = torch.squeeze(torch.from_numpy(self.gene_exps_1h[gene]).float())
        if self.ptb_type == 'onehot':
            return p1h[:-1], p1h
        elif self.ptb_type == 'expression':
            p = torch.squeeze(torch.from_numpy(self.gene_exps[gene]).float())
            return p[np.random.choice(p.shape[0], self.ptb_dim)], p1h
        assert self.ptb_type == 'pca'
        p = torch.squeeze(torch.from_numpy(self.pca_embeddings[gene])).float()
        return p, p1h

class SCDATA_sampler(Sampler):
    def __init__(self, scdataset, batchsize):
        self.intervindices = []
        self.len = 0
        ptb_name = scdataset.get_geneslist()
        for ptb in set(ptb_name):
            idx = np.where(scdataset.data['ptb'] == ptb)[0]
            self.intervindices.append(idx)
            self.len += len(idx) // batchsize
        self.batchsize = batchsize
        shuffle = (scdataset.mode != 'test') # shuffle is false if test else true
        self.combined = self.make_batches(shuffle)
 
    def chunk(self, indices, chunk_size):
        split = torch.split(torch.tensor(indices), chunk_size)
        
        if len(indices) % chunk_size == 0:
            return split
        elif len(split) > 0:
            return split[:-1]
        else:
            return None
    
    def make_batches(self, shuffle = True):
        print('making batches')
        comb = []
        for i in range(len(self.intervindices)):
            if shuffle:
                random.shuffle(self.intervindices[i])
        
            interv_batches = self.chunk(self.intervindices[i], self.batchsize)
            if interv_batches:
                comb += interv_batches

            combined = [batch.tolist() for batch in comb]
            if shuffle:
                random.shuffle(combined)
        return combined
    
    def __iter__(self):
        return iter(self.combined)
    
    def __len__(self):
        return self.len