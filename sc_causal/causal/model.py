import torch
import torch.nn as nn
import networkx as nx
from math import log, pi
import numpy as np
import scanpy as sc
from torch.distributions import Normal

from torch.distributions import Gamma, Poisson
import torch.nn.functional as F
from torch.distributions import kl_divergence as kl

import sys
sys.path.append('./')

from causal.modules import Encoder, PTBEncoder, Decoder_Gaussian, MMD_loss

class CausalVAE_Gaussian(nn.Module):
    def __init__(self, exp_dim, ptb_dim, z_dim, enc_hidden, dec_hidden, B_filename, mode, device, ptb_encode, rand_graph_seed = 0):
        torch.set_default_dtype(torch.float64)
        super(CausalVAE_Gaussian, self).__init__()

        self.mode = mode

        self.device = torch.device(device)
        self.z_encoder = Encoder(input_dim=exp_dim, z_dim=z_dim, hidden_dims=enc_hidden)
        self.perturbation_encoder = PTBEncoder(input_dim=ptb_dim, out_dim=z_dim, hidden_dims=enc_hidden[:2])

        if ptb_encode == 'genept_finetune':
            self.perturbation_encoder.load_state_dict(torch.load('./train_util_files/pretrained_genept.pth'))
        elif ptb_encode == 'onehot_finetune':
            self.perturbation_encoder.load_state_dict(torch.load('./train_util_files/pretrained_onehot.pth'))
        elif ptb_encode == 'pca_finetune':
            self.perturbation_encoder.load_state_dict(torch.load('./train_util_files/pretrained_pca.pth'))
        elif ptb_encode == 'expression_finetune':
            self.perturbation_encoder.load_state_dict(torch.load('./train_util_files/pretrained_expression.pth'))
        else:
            assert ptb_encode in {'pca', 'genept', 'onehot', 'expression'}, 'Invalid ptb_encode argument. Must be one of genept, onehot, pca, or expression.'

        decoder_input_dim = z_dim if mode != 'cvae' else z_dim + ptb_dim

        self.decoder = Decoder_Gaussian(z_dim=decoder_input_dim, output_dim = exp_dim, hidden_dims=dec_hidden)

        self.encode_ptb = (ptb_encode is not None) # if we use a perturbation encoding, set to True. Else, set to False

        self.zdim = z_dim
        self.ptb_dim = ptb_dim

        with open(B_filename, 'rb') as f:
            B_mask = np.load(f)
        
        self.B_mask = torch.from_numpy(B_mask).to(self.device)

        if self.mode == 'random':
            ratio = 2*np.sum(B_mask) / (B_mask.shape[0] * B_mask.shape[1])
            B_mask = nx.fast_gnp_random_graph(self.zdim, ratio, seed=rand_graph_seed)
            B_mask = nx.to_numpy_array(B_mask)
            B_mask = np.triu(B_mask)
            self.B_mask = torch.from_numpy(B_mask).to(self.device)
        elif self.mode == 'conditional':
            self.B_mask = torch.zeros(B_mask.shape).to(self.device)
        elif self.mode == 'full':
            self.B_mask = torch.ones(B_mask.shape).to(self.device)
            self.B_mask = torch.triu(self.B_mask, diagonal=1)
        
        self.state_dict()['B_mask'] = self.B_mask

        self.B = torch.nn.Parameter(torch.normal(0,.1,size = (self.zdim, self.zdim)))
        self.B.requires_grad = True

        self.s = nn.Parameter(torch.ones(self.zdim))
        self.s.requires_grad = True

        full_adata = sc.read_h5ad('./h5ad_datafiles/k562_annotated_raw.h5ad')

        sc.pp.normalize_total(full_adata)
        sc.pp.log1p(full_adata)
        nontargeting = full_adata[full_adata.obs['gene'] == 'non-targeting']
        
        self.Xs = nontargeting.X
        del full_adata
        del nontargeting

        self.X_means = torch.mean(torch.from_numpy(self.Xs), axis=0).to(self.device)

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus()
        self.mse_loss = torch.nn.MSELoss()
        self.mmd_loss = MMD_loss(fix_sigma=1000, kernel_num=10)

        for name, param in self.named_parameters():
            assert param.requires_grad, f'{name} is not set to require grad'

    def compute_latents(self, y, p, p1h):
        qz_dist, z = self.z_encoder(y)

        ptb_enc = torch.zeros_like(z)

        if self.encode_ptb:
            ptb_enc = self.perturbation_encoder(p)
        else:
            ptb_enc = p1h[:,:-1]

        if self.mode == 'cvae':
            ptb_enc = p

        return {
            'z': z,
            'qz_dist': qz_dist,
            'ptb_enc': ptb_enc, 
            'c': p1h[:,:-1] @ self.s
        }
    
    def reconstruct(self, z, p_enc, c = None):
        if c is None:
            c = self.s
        if self.mode in ['causal', 'random', 'full']:
            c = c.reshape(-1, 1)
            shift = z + c*p_enc
            scale = torch.inverse(torch.eye(self.zdim).to(self.device) - self.B*self.B_mask) # zdim by zdim
            u = torch.matmul(shift.double(), scale.double())
            u0 = torch.matmul(z.double(), scale.double())
        elif self.mode == 'conditional':
            c = c.reshape(-1, 1)
            u = z + c*p_enc
            u0 = z

        out = self.decoder(u)
        out_X = self.decoder(u0)

        return {
            'X': out_X,
            'u': u,
            'y': out,
            'shift': c*p_enc,
        }

    def forward(self, y, p, p1h, gene=None, train_hard = False):
        if train_hard:
            X_sample = self.Xs[np.random.choice(self.Xs.shape[0], y.shape[0]),:]
            X_sample = torch.from_numpy(X_sample).to(self.device).double()
            latents = self.compute_latents(X_sample, p, p1h, gene=gene)
        else:
            latents = self.compute_latents(y, p, p1h, gene=gene)

        recons = self.reconstruct(latents['z'], latents['ptb_enc'], latents['c'])

        pz_dist = Normal(torch.zeros_like(latents['z']), torch.ones_like(latents['z']))

        return {
            'qz': latents['qz_dist'],
            'pz': pz_dist,
            'y': recons['y'],
            'X_pred': recons['X'],
            'X_gt': X_sample if train_hard else y
        }
 
    def loss_function(self, y, outs, beta, alpha, recon_scale = 1.0):
        kl_divergence_z = kl(outs["qz"], outs["pz"]).mean(dim = -1)

        reconst_loss_X = self.mse_loss(outs['X_pred'], outs['X_gt'])
        reconst_loss = self.mse_loss(outs['y'], y)
        mmd_loss = self.mmd_loss(outs['y'], y)

        weighted_kl = beta * kl_divergence_z
        weighted_mmd = alpha * mmd_loss

        direction_lambda = 1
        dir_loss = torch.mean(direction_lambda * (torch.sign(y - self.X_means) - torch.sign(outs['y'] - self.X_means))**2)

        loss = torch.mean(recon_scale * reconst_loss_X + reconst_loss + weighted_kl + weighted_mmd)

        loss_dict = {
            'recons_X': torch.mean(reconst_loss_X.detach()), 
            'recons': torch.mean(reconst_loss.detach()),
            'kl':torch.mean(kl_divergence_z.detach()),  
            'mmd': torch.mean(mmd_loss.detach()),
            'dir_loss': torch.mean(dir_loss.detach()),
            'loss': loss
        }

        return loss_dict

    def step(self, y, p, p1h, gene, optimizer, beta, alpha, train_hard, recon_scale, clip_grad = True):
        assert self.training is True

        outputs = self.forward(y, p, p1h, gene, train_hard)
        losses_dict = self.loss_function(y, outputs, beta, alpha, recon_scale)
        loss = losses_dict['loss']

        optimizer.zero_grad()
        loss.backward()

        if clip_grad:
            for _, param in self.named_parameters():
                if param.requires_grad:
                    torch.nn.utils.clip_grad_norm_(param, 1.0)
        
        optimizer.step()

        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(name)
                print(param)
                assert False

        return loss.item(), losses_dict

    def save(self, path, optimizer=None, scheduler=None):
        state_dicts = {
            'model_state': self.state_dict()
        }

        if optimizer is not None:
            state_dicts['optimizer_state'] = optimizer.state_dict()
        
        if scheduler is not None:
            state_dicts['scheduler_state'] = scheduler.state_dict()

        torch.save(state_dicts, path)
