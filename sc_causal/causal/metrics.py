import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from causal.dataset import PerturbDataset, SCDATA_sampler
from math import ceil
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

## CODE FOR GENERATING MARKER GENES

def generate_marker_genes():
    adata = sc.read_h5ad('h5ad_datafiles/k562_annotated_raw.h5ad')
    adata = adata[adata.obs['gene'] == 'non-targeting']
    sc.pp.log1p(adata)
    sc.tl.rank_genes_groups(adata, groupby='gene', rankby_abs=True)
    top50 = [x[0] for x in list(adata.uns['rank_genes_groups']['names'][:50])]
    varnames = list(adata.var_names)
    top_50_marker = []
    for gene in top50:
       top_50_marker.append(varnames.index(gene))
    return top_50_marker

# The above code generates the top 50 marker genes for the non-targeting perturbation
top_50_marker = [8562, 2858, 2844, 2845, 2846, 2847, 2848, 2849, 2850, 2851, 2852, 2853, 2854, 2855, 2856, 2857, 2859, 
                 2875, 2860, 2861, 2862, 2863, 2864, 2865, 2866, 2867, 2868, 2869, 2870, 2871, 2872, 2873, 2843, 2842, 
                 2841, 2840, 2811, 2812, 2813, 2814, 2815, 2816, 2817, 2818, 2819, 2820, 2821, 2822, 2823, 2824]

def umap(adata):
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    return adata


def pca(adata):
    sc.tl.pca(adata, svd_solver='arpack')
    return adata

# Take average of all expressions per perturbation, or average of everything if None, and return MSE
# ptb is either None or a single perturbation
def mse(adata, ptb=None, marker_genes = False):
    if ptb is not None:
        adata = adata[adata.obs['ptb'].isin(['nontargeting_ref', ptb])]
    adata_predicted = adata[adata.obs['y'] == 'predicted']
    adata_true = adata[adata.obs['y'] == 'ground truth']
    if adata_predicted.shape[0] == 0 or adata_true.shape[0] == 0:
        return None
    
    if len(adata_predicted) != len(adata_true):
        adata_predicted = adata_predicted[np.random.choice(adata_predicted.shape[0], adata_true.shape[0])]

    if marker_genes:
        adata_predicted = adata_predicted[:, top_50_marker]
        adata_true = adata_true[:, top_50_marker]

    return np.mean((adata_predicted.X.mean(0) - adata_true.X.mean(0))**2)

def mse_per_ptb(adata, ptbs=None, marker_genes = False):
    if ptbs is None:
        ptbs = sorted(list(set(adata.obs['ptb'])))
    
    mses = {}
    for one_ptb in ptbs:
        val = mse(adata, one_ptb, marker_genes)
        if val is not None:
            mses[one_ptb] = val
    return mses

def energy_distance(adata, ptb=None, marker_genes = False):
    if ptb is not None:
        adata = adata[adata.obs['ptb'].isin(['nontargeting_ref', ptb])]
    adata_predicted = adata[adata.obs['y'] == 'predicted']
    adata_true = adata[adata.obs['y'] == 'ground truth']
    if adata_predicted.shape[0] == 0 or adata_true.shape[0] == 0:
        return None
    
    if len(adata_predicted) != len(adata_true):
        adata_predicted = adata_predicted[np.random.choice(adata_predicted.shape[0], adata_true.shape[0])]

    if marker_genes:
        adata_predicted = adata_predicted[:, top_50_marker]
        adata_true = adata_true[:, top_50_marker]

    batch_size = 32
    losses = []

    indices = [x for x in range(len(adata_predicted))]
    adata_predicted.obs['idx'] = indices
    adata_true.obs['idx'] = indices

    for i in range(len(indices) // batch_size):
        preds = adata_predicted[i*batch_size:(i+1)*batch_size, :].X
        gt = adata_true[i*batch_size:(i+1)*batch_size, :].X

        preds_perm = preds[np.random.permutation(preds.shape[0])]
        gt_perm = gt[np.random.permutation(gt.shape[0])]

        dXY = ((preds - gt)**2).mean()
        dXX = ((preds - preds_perm)**2).mean()
        dYY = ((gt - gt_perm)**2).mean()

        loss = 2*dXY - dXX - dYY
        losses.append(loss)
    
    losses = np.array(losses)

    return np.mean(losses)


def energy_per_ptb(adata, ptbs=None, marker_genes = False):
    if ptbs is None:
        ptbs = sorted(list(set(adata.obs['ptb'])))
    
    mses = {}
    for one_ptb in ptbs:
        val = energy_distance(adata, one_ptb, marker_genes)
        if val is not None:
            mses[one_ptb] = val
    return mses


def mmd(adata, ptb, marker_genes = False, baseline = False):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=1000):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    
    adata = adata[adata.obs['ptb'].isin([ptb, 'nontargeting_ref'])]
    adata_predicted = adata[adata.obs['y'] == 'predicted']
    adata_true = adata[adata.obs['y'] == 'ground truth']

    if adata_predicted.shape[0] == 0 or adata_true.shape[0] == 0:
        return None, None

    if len(adata_predicted) != len(adata_true):
        adata_predicted = adata_predicted[np.random.choice(adata_predicted.shape[0], adata_true.shape[0])]

    if marker_genes:
        adata_predicted = adata_predicted[:, top_50_marker]
        adata_true = adata_true[:, top_50_marker]

    batch_size = 32

    losses = []

    indices = [x for x in range(len(adata_predicted))]
    adata_predicted.obs['idx'] = indices
    adata_true.obs['idx'] = indices

    for i in range(len(indices) // batch_size):
        preds = adata_predicted[i*batch_size:(i+1)*batch_size, :]
        gt = adata_true[i*batch_size:(i+1)*batch_size, :]

        kernels = gaussian_kernel(
            torch.from_numpy(preds.X).to(device), 
            torch.from_numpy(gt.X).to(device), 
            kernel_mul=2.0, kernel_num=5, fix_sigma=None)

        n_samples = preds.shape[0]
        
        XX = kernels[:n_samples, :n_samples]
        YY = kernels[n_samples:, n_samples:]
        XY = kernels[:n_samples, n_samples:]
        YX = kernels[n_samples:, :n_samples]

        loss = torch.mean(XX + YY - XY -YX).item()
        losses.append(loss)
    
    losses = np.array(losses)

    return np.mean(losses), np.std(losses)


def mmd_per_ptb(adata, ptbs=None, marker_genes = False, flag=True, baseline=False):
    if ptbs is None:
        ptbs = sorted(list(set(adata.obs['ptb'])))
    mmds = {}
    for one_ptb in ptbs:
        if flag:
            mmd_mean, mmd_std = mmd(adata, one_ptb, marker_genes, baseline)
            if mmd_mean is None:
                continue
            mmds[one_ptb] = mmd_mean
        else:
            mmds[one_ptb] = mmd_cellot(adata, one_ptb, marker_genes, baseline)
    return mmds


def predict_pearsonr(adata, ptb=None, marker_genes = False):
    if ptb is not None:
        adata = adata[adata.obs['ptb'].isin(['nontargeting_ref', ptb])]
    adata_predicted = adata[adata.obs['y'] == 'predicted'].X
    adata_true = adata[adata.obs['y'] == 'ground truth'].X

    if marker_genes:
        adata_predicted = adata_predicted[:, top_50_marker]
        adata_true = adata_true[:, top_50_marker]
    

    if adata_predicted.shape[0] == 0 or adata_true.shape[0] == 0:
        return None
    if len(adata_predicted) != len(adata_true):
        adata_predicted = adata_predicted[np.random.choice(adata_predicted.shape[0], adata_true.shape[0])]
    

    adata_true = adata_true.mean(0)
    adata_predicted = adata_predicted.mean(0)

    return pearsonr(adata_true, adata_predicted)[0]

def pearsonr_per_ptb(adata, ptbs=None, marker_genes = False, baseline = False):
    if ptbs is None:
        ptbs = sorted(list(set(adata.obs['ptb'])))

    r2s = {}
    
    for one_ptb in ptbs:
        val = predict_pearsonr(adata, one_ptb, marker_genes)
        if val is not None:
            r2s[one_ptb] = val
    
    return r2s

def direction_of_change(adata, ctrl, ptb=None, marker_genes = False):
    if ptb is not None:
        adata = adata[adata.obs['ptb'].isin(['nontargeting_ref', ptb])]
    adata_predicted = adata[adata.obs['y'] == 'predicted'].X
    adata_true = adata[adata.obs['y'] == 'ground truth'].X

    if adata_predicted.shape[0] == 0 or adata_true.shape[0] == 0:
        return None, None, None

    direc_change = np.abs(np.sign(adata_predicted.mean(0) - ctrl) - np.sign(adata_true.mean(0) - ctrl))  
    frac_correct_direction = len(np.where(direc_change == 0)[0])/adata_predicted.shape[1]
    frac_01_direction = len(np.where(direc_change == 1)[0])/adata_predicted.shape[1]
    frac_inc_direction = len(np.where(direc_change == 2)[0])/adata_predicted.shape[1]

    return frac_correct_direction, frac_01_direction, frac_inc_direction

def frac_changed_direction_per_ptb(adata, ptbs=None, marker_genes = False):
    if ptbs is None:
        ptbs = sorted(list(set(adata.obs['ptb'])))
    
    if marker_genes:
        adata = adata[:, top_50_marker]
    
    if len(adata[adata.obs['mode'] == 'nontargeting']) > 0:
        ctrl = adata[adata.obs['mode'] == 'nontargeting'].X.mean(0)
    elif 'non-targeting' in set(adata[adata.obs['y'] == 'ground truth'].obs['ptb']):
        adata_gt = adata[adata.obs['y'] == 'ground truth']
        ctrl = adata_gt[adata_gt.obs['ptb'] == 'non-targeting'].X.mean(0)
    else:
        assert len(adata[adata.obs['ptb'] == 'nontargeting_ref']) > 0
        ctrl = adata[adata.obs['ptb'] == 'nontargeting_ref'].X.mean(0)

    fracs = {}
    for one_ptb in ptbs:
        dirs = direction_of_change(adata, ctrl, one_ptb, marker_genes)[2]
        if dirs is not None:
            fracs[one_ptb] = dirs
    
    return fracs


def frac_correct_direction_per_ptb(adata, ptbs=None, marker_genes = False):
    if ptbs is None:
        ptbs = sorted(list(set(adata.obs['ptb'])))
    
    if marker_genes:
        adata = adata[:, top_50_marker]
    
    if len(adata[adata.obs['mode'] == 'nontargeting']) > 0:
        ctrl = adata[adata.obs['mode'] == 'nontargeting'].X.mean(0)
    elif 'non-targeting' in set(adata[adata.obs['y'] == 'ground truth'].obs['ptb']):
        adata_gt = adata[adata.obs['y'] == 'ground truth']
        ctrl = adata_gt[adata_gt.obs['ptb'] == 'non-targeting'].X.mean(0)
    else:
        assert len(adata[adata.obs['ptb'] == 'nontargeting_ref']) > 0
        ctrl = adata[adata.obs['ptb'] == 'nontargeting_ref'].X.mean(0)

    fracs = {}
    for one_ptb in ptbs:
        dirs = direction_of_change(adata, ctrl, one_ptb, marker_genes)[0]
        if dirs is not None:
            fracs[one_ptb] = dirs
    
    return fracs