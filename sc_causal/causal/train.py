import numpy as np
import torch
from tqdm import tqdm
import os
import wandb

import sys
sys.path.append('./')

from causal.model import CausalVAE_Gaussian as CausalVAE

def train(
        device, 
        dataloaders, 
        datasets,
        learning_rate, 
        num_epochs, 
        beta_start, 
        beta_max, 
        gamma_start,
        gamma_max,
        savedir, 
        hidden_size, 
        n_hidden, 
        beta_end, 
        ptb_dim,
        mode, 
        schedule = None,
        pretrain_dir = None, 
        ptb_encode = None,
        marker = False,
        train_hard = True,
        recon_scale = 0.25,
        rand_graph_seed = 0,
        parent_dir = False):
    torch.set_default_dtype(torch.float64)

    prefix = '.' if parent_dir else ''

    run = wandb.init(project='crl', name=savedir, dir=savedir)
    net = CausalVAE(
        ptb_dim=ptb_dim,
        exp_dim = 8563,
        z_dim = 512,
        enc_hidden = [hidden_size]*n_hidden,
        dec_hidden = [hidden_size]*(n_hidden),
        B_filename=f'{prefix}./train_util_files/B_512_upper_triangular.npy',
        device = device,
        mode = mode,
        ptb_encode = ptb_encode,
        rand_graph_seed=rand_graph_seed,
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, eps=0.01, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = schedule['factor'], patience=schedule['patience']) if schedule else None

    wandb.watch(net, log_freq=100, log = 'all')

    if pretrain_dir is not None: # Finetuning for OOD tasks possibly
        checkpoint = torch.load(pretrain_dir)
        net.load_state_dict(checkpoint['model_state'])

        for param in net.z_encoder.parameters():
            param.requires_grad = False
        for param in net.decoder.parameters():
            param.requires_grad = False
        
        net.B.requires_grad = False


    net.to(device)

    best_train_loss = np.inf
    best_train_mmd_loss = np.inf
    best_train_mmd_recon_loss = np.inf
    best_val_loss = np.inf
    best_val_mmd_loss = np.inf
    best_val_mmd_recon_loss = np.inf

    if beta_start != -1:
        beta_schedule = torch.zeros(num_epochs)
        beta_schedule[:beta_start] = 0
        if beta_end == -1:
            beta_end = num_epochs-beta_start
        beta_schedule[beta_start:beta_end] = torch.linspace(0, beta_max, beta_end - beta_start)
    else:
        beta_schedule = torch.ones(num_epochs) * beta_max
    
    gamma_schedule = torch.ones(num_epochs) * gamma_max
    
    phases = ['train', 'val'] if 'val' in dataloaders else ['train']

    best_train_epoch = -1
    best_train_mmd_epoch = -1
    best_train_mmd_recon_epoch = -1
    
    best_val_epoch = -1
    best_val_mmd_epoch = -1
    best_val_mmd_recon_epoch = -1

    
    for epoch in range(num_epochs):
        log = {}
        for phase in phases:
            running_losses = {}
            num_preds = 0

            bar = tqdm(dataloaders[phase], desc='CausalVAE Epoch {} {}'.format(epoch, phase).ljust(20))
            for i, batch in enumerate(bar):
                y, p, p1h, gene = batch
                gene = gene[0]
                y, p, p1h = y.to(device), p.to(device), p1h.to(device)

                if phase == 'train':
                    loss, losses_dict = net.step(y, p, p1h, gene, optimizer, beta_schedule[epoch], gamma_schedule[epoch], train_hard, recon_scale)
                else:
                    outs = net(y, p, p1h)
                    losses_dict = net.loss_function(y, outs, beta_schedule[epoch], gamma_schedule[epoch], recon_scale)
                    loss = losses_dict['loss'].item()

                for elem in losses_dict:
                    if elem not in running_losses:
                        running_losses[elem] = 0
                    running_losses[elem] += losses_dict[elem].item()

                num_preds += 1

                if i % 10 == 0:
                    bar.set_postfix(loss=f'{loss}')
            
            for elem in running_losses:
                running_losses[elem] /= num_preds
                log[f'{elem} {phase}'] = running_losses[elem]
            
            epoch_elbo = running_losses['recons'] + running_losses['kl']
            epoch_elbo_mmd = running_losses['mmd'] + running_losses['kl']
            epoch_elbo_mmd_recon = running_losses['mmd'] + running_losses['kl'] + running_losses['recons']

            log[f'elbo {phase}'] = epoch_elbo
            log[f'elbo_mmd {phase}'] = epoch_elbo_mmd
            log[f'elbo_mmd_recon {phase}'] = epoch_elbo_mmd_recon
            
            if scheduler is not None:
                scheduler.step(epoch_elbo)
            
            if phase == 'train':
                if epoch_elbo < best_train_loss:
                    best_train_epoch = epoch
                    best_train_loss = epoch_elbo
                    net.save(os.path.join(savedir, 'best_train.pth'), optimizer=optimizer, scheduler=scheduler)
                if epoch_elbo_mmd < best_train_mmd_loss:
                    best_train_mmd_epoch = epoch
                    best_train_mmd_loss = epoch_elbo_mmd
                    net.save(os.path.join(savedir, 'best_train_mmd.pth'), optimizer=optimizer, scheduler=scheduler)
                if epoch_elbo_mmd_recon < best_train_mmd_recon_loss:
                    best_train_mmd_recon_epoch = epoch
                    best_train_mmd_recon_loss = epoch_elbo_mmd_recon
                    net.save(os.path.join(savedir, 'best_train_mmd_recon.pth'), optimizer=optimizer, scheduler=scheduler)
                if (epoch + 1) % 10 == 0:
                    net.save(os.path.join(savedir, f'checkpoint_{epoch+1}.pth'), optimizer=optimizer, scheduler=scheduler)
            if phase == 'val':
                if epoch_elbo < best_val_loss:
                    best_val_epoch = epoch
                    best_val_loss = epoch_elbo
                    net.save(os.path.join(savedir, 'best_val.pth'), optimizer=optimizer, scheduler=scheduler)
                if epoch_elbo_mmd < best_val_mmd_loss:
                    best_val_mmd_epoch = epoch
                    best_val_mmd_loss = epoch_elbo_mmd
                    net.save(os.path.join(savedir, 'best_val_mmd.pth'), optimizer=optimizer, scheduler=scheduler)
                if epoch_elbo_mmd_recon < best_val_mmd_recon_loss:
                    best_val_mmd_recon_epoch = epoch
                    best_val_mmd_recon_loss = epoch_elbo_mmd_recon
                    net.save(os.path.join(savedir, 'best_val_mmd_recon.pth'), optimizer=optimizer, scheduler=scheduler)

            wandb.log(log)
    run.finish() 
    print(f'Best train epoch: {best_train_epoch}')
    print(f'Best val epoch: {best_val_epoch}')
    print(f'Best train mmd epoch: {best_train_mmd_epoch}')
    print(f'Best val mmd epoch: {best_val_mmd_epoch}')
    print(f'Best train mmd recon epoch: {best_train_mmd_recon_epoch}')
    print(f'Best val mmd recon epoch: {best_val_mmd_recon_epoch}')

    with open(f'{savedir}/besttraintest.txt', 'w') as f:
        f.write(f'Best train epoch: {best_train_epoch}\n')
        f.write(f'Best val epoch: {best_val_epoch}\n')
        f.write(f'Best train mmd epoch: {best_train_mmd_epoch}\n')
        f.write(f'Best val mmd epoch: {best_val_mmd_epoch}\n')
        f.write(f'Best train mmd recon epoch: {best_train_mmd_recon_epoch}\n')
        f.write(f'Best val mmd recon epoch: {best_val_mmd_recon_epoch}\n')
    
    net.save(os.path.join(savedir, 'last_model.pth'), optimizer=optimizer, scheduler=scheduler)

    return net
