import sys
import anndata as ad
sys.path.append('../')
import numpy as np

from gears import PertData, GEARS
from gears.inference import evaluate, compute_metrics
import scanpy as sc

import argparse

def train(id):
    pert_data = PertData('./data')
    if id == -1:
        pert_data.load(data_path = './data/ours_all')
    else:
        pert_data.load(data_path = f'./data/ours_ood_split_{id}')
    
    pert_data.prepare_split(split = 'no_test', seed = 1)
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)


    gears_model = GEARS(pert_data, device = f'cuda:0', 
                            weight_bias_track = False, 
                            proj_name = 'pertnet', 
                            exp_name = 'pertnet')
    
    gears_model.model_initialize(hidden_size = 64)

    print(gears_model.tunable_parameters())

    gears_model.train(epochs = 20, lr = 1e-3)

    if id >= 0:
        gears_model.save_model(f'test_model_ood_{id}')
    else:
        assert id == -1
        gears_model.save_model(f'test_model')


def inference(id):
    pert_data = PertData('./data')
    if id == -1:
        pert_data.load(data_path = './data/ours_all_test')
    else:
        pert_data.load(data_path = f'./data/ours_ood_split_{id}_test')
    pert_data.prepare_split(split = 'no_split', seed = 1)
    pert_data.dataloader = pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)


    gears_eval = GEARS(pert_data, device = f'cuda:0', 
                            weight_bias_track = False, 
                            proj_name = 'pertnet', 
                            exp_name = 'pertnet')
    gears_eval.load_pretrained(f'test_model_ood_{id}')

    results = evaluate(pert_data.dataloader['test_loader'], gears_eval.model, False, f'cuda:0')
    

    preds = ad.AnnData(results['pred'])
    preds.obs['y'] = ['predicted'] * preds.shape[0]
    preds.obs['ptb'] = [x.split('+')[0] for x in results['pert_cat']]
    gt = ad.AnnData(results['truth'])
    gt.obs['y'] = ['ground truth'] * gt.shape[0]
    gt.obs['ptb'] = [x.split('+')[0] for x in results['pert_cat']]

    adata = gt.concatenate(preds)

    adata_og = sc.read_h5ad('../../h5ad_datafiles/k562_annotated_raw_allnontargetinginval.h5ad')

    adata_og = adata_og[adata_og.obs['gene'] == 'non-targeting']
    sc.pp.normalize_total(adata_og)
    sc.pp.log1p(adata_og)
    adata_og = adata_og[adata_og.obs['split'] == 'train'].X
    nontargeting = ad.AnnData(adata_og)
    nontargeting.obs['mode'] = ['nontargeting']*nontargeting.shape[0]
    nontargeting.obs['ptb'] = ['nontargeting_ref']*nontargeting.shape[0]
    nontargeting.obs['y'] = ['nontargeting_ref']*nontargeting.shape[0]

    adata.obs['mode'] = ['gears']*adata.shape[0]
    adata = ad.concat([adata, nontargeting])

    adata.write(f'inference_anndatas/results_gears_ood_split_{id}_withnontargeting.h5ad')


def metrics():
    sys.path.append('../../')
    from causal.metrics import mse_per_ptb, mmd_per_ptb, pearsonr_per_ptb, frac_correct_direction_per_ptb, frac_changed_direction_per_ptb, energy_per_ptb
    from causal.inference import derive_nontargeting_from_adata

    fnames = [
        'results_gears_withnontargeting.h5ad',
        'results_gears_ood_split_0_withnontargeting.h5ad',
        'results_gears_ood_split_1_withnontargeting.h5ad',
        'results_gears_ood_split_2_withnontargeting.h5ad',
        'results_gears_ood_split_3_withnontargeting.h5ad',
        'results_gears_ood_split_4_withnontargeting.h5ad'
    ]

    for fname in fnames:
        print(fname)
        adata = ad.read('inference_anndatas/'+fname)

        excl = {'nontargeting', 'INTS2', 'POLR2B', 'SMN2', 'DDX47', 'TREX2', 'PHB', 'BGLAP', 'ANAPC2', 'ZNF687'}
        ptbs = [x for x in list(set(adata.obs['ptb'])) if not (x in excl)]

        print('GEARS')
        precision = 3

        for top50, lbl in zip([False, True], ['All genes', 'Top 50']):
            mses_dict = mse_per_ptb(adata, ptbs=ptbs, marker_genes=top50)
            mses = np.array([x for x in mses_dict.values()])
            print(f'MSE {lbl}: {np.mean(mses):.{precision}} $\pm$ {np.std(mses):.{precision}}')

            mmds_dict = mmd_per_ptb(adata, ptbs=ptbs, marker_genes=top50)
            mmds = np.array([x for x in mmds_dict.values()])
            print(f'MMD {lbl}: {np.mean(mmds):.{precision}} $\pm$ {np.std(mmds):.{precision}}')

            pearsonr_dict = pearsonr_per_ptb(adata, ptbs=ptbs, marker_genes=top50)
            pearsons = np.array([x for x in pearsonr_dict.values()])
            print(f'Pearsons {lbl}: {np.mean(pearsons):.{precision}} $\pm$ {np.std(pearsons):.{precision}}')
            print('Pearson', lbl, ':', np.mean(pearsons), '+-', np.std(pearsons))

            frac_same_dict = frac_correct_direction_per_ptb(adata, ptbs=ptbs, marker_genes=top50)
            frac_same = np.array([x for x in frac_same_dict.values()])
            print(f'Frac Same Direction {lbl}: {np.mean(frac_same):.{precision}} $\pm$ {np.std(frac_same):.{precision}}')

            frac_changed_dict = frac_changed_direction_per_ptb(adata, ptbs=ptbs, marker_genes=top50)
            frac_changed = np.array([x for x in frac_changed_dict.values()])
            print(f'Frac Changed Direction {lbl}: {np.mean(frac_changed):.{precision}} $\pm$ {np.std(frac_changed):.{precision}}')

            energy_dict = energy_per_ptb(adata, ptbs=ptbs, marker_genes=top50)
            energy = np.array([x for x in energy_dict.values()])
            print(f'Energy {lbl}: {np.mean(energy):.{precision}} $\pm$ {np.std(energy):.{precision}}')
        

        nontargeting = derive_nontargeting_from_adata(adata[adata.obs['y'].isin(['ground truth', 'nontargeting_ref'])])
        print('NONTARGETING')
        for top50, lbl in zip([False, True], ['All genes', 'Top 50']):

            mses_dict = mse_per_ptb(nontargeting, ptbs=ptbs, marker_genes=top50)
            mses = np.array([x for x in mses_dict.values()])
            print(f'MSE {lbl}: {np.mean(mses):.{precision}} $\pm$ {np.std(mses):.{precision}}')

            mmds_dict = mmd_per_ptb(nontargeting, ptbs=ptbs, marker_genes=top50)
            mmds = np.array([x for x in mmds_dict.values()])
            print(f'MMD {lbl}: {np.mean(mmds):.{precision}} $\pm$ {np.std(mmds):.{precision}}')

            pearsonr_dict = pearsonr_per_ptb(nontargeting, ptbs=ptbs, marker_genes=top50)
            pearsons = np.array([x for x in pearsonr_dict.values()])
            print(f'Pearsons {lbl}: {np.mean(pearsons):.{precision}} $\pm$ {np.std(pearsons):.{precision}}')
            print('Pearson', lbl, ':', np.mean(pearsons), '+-', np.std(pearsons))

            frac_same_dict = frac_correct_direction_per_ptb(nontargeting, ptbs=ptbs, marker_genes=top50)
            frac_same = np.array([x for x in frac_same_dict.values()])
            print(f'Frac Same Direction {lbl}: {np.mean(frac_same):.{precision}} $\pm$ {np.std(frac_same):.{precision}}')

            frac_changed_dict = frac_changed_direction_per_ptb(nontargeting, ptbs=ptbs, marker_genes=top50)
            frac_changed = np.array([x for x in frac_changed_dict.values()])
            print(f'Frac Changed Direction {lbl}: {np.mean(frac_changed):.{precision}} $\pm$ {np.std(frac_changed):.{precision}}')

            energy_dict = energy_per_ptb(nontargeting, ptbs=ptbs, marker_genes=top50)
            energy = np.array([x for x in energy_dict.values()])
            print(f'Energy {lbl}: {np.mean(energy):.{precision}} $\pm$ {np.std(energy):.{precision}}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)

    args = parser.parse_args()

    id = args.id
    train(id=id)
    inference(id=id)

    metrics()