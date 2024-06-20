import os
import argparse
import time
import numpy as np
from pathlib import Path
import cv2
import itertools
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from copy import deepcopy
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from utils.kari_img_trans_dataset import KariImgTransDataset
from models.gans import Generator, Discriminator
from utils.utils import intersect_dicts, AverageMeter, add_module_prefix
from utils.torch_utils import de_parallel, is_main_process

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))   # https://pytorch.org/docs/stable/elastic/run.html

def train(opt):
    epochs, batch_size, name, = opt.epochs, opt.batch_size, opt.name
    # wandb settings
    wandb.init(id=opt.name, resume='allow')
    wandb.config.update(opt)
    
    train_dataset = KariImgTransDataset(root='data/wv3-k3a/train', train=True)
    num_workers = min([os.cpu_count(), batch_size])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    val_dataset = KariImgTransDataset(root='data/wv3-k3a/train', train=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    
    # GPU-support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device.type != 'cpu'
    
    # model
    G_XY = Generator().to(device)
    G_YX = Generator().to(device)
    D_X = Discriminator().to(device)
    D_Y = Discriminator().to(device)
    
    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        if is_main_process():
            print("DP mode is enabled, but DDP is preferred for best performance.")
        G_XY = torch.nn.DataParallel(G_XY)
        G_YX = torch.nn.DataParallel(G_YX)
        D_X = torch.nn.DataParallel(D_X)
        D_Y = torch.nn.DataParallel(D_Y)
        
    # DDP mode
    if cuda and RANK != -1:
        G_XY = DDP(G_XY, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        G_YX = DDP(G_YX, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        D_X = DDP(D_X, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
        D_Y = DDP(D_Y, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    
    # optimizer
    G_optimizer = optim.Adam(itertools.chain(G_XY.parameters(), G_YX.parameters()), lr=2e-4, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(itertools.chain(D_X.parameters(), D_Y.parameters()), lr=2e-4, betas=(0.5, 0.999))

    # scaler
    D_scaler = amp.GradScaler(enabled=cuda)
    G_scaler = amp.GradScaler(enabled=cuda)

    # loading a weight file (if exists)
    weight_file = Path('weights') / (name + '_best.pth')
    lowest_loss = 100
    start_epoch, end_epoch = (0, epochs)
    if os.path.exists(weight_file):
        print('loading the best model...')
        ckpt = torch.load(weight_file, map_location='cpu')
        models_and_keys = [
            (G_XY, 'G_XY'),
            (G_YX, 'G_YX'),
            (D_X, 'D_X'),
            (D_Y, 'D_Y')
        ]
        for model, key in models_and_keys:
            csd = ckpt[key].float().state_dict()
            csd = intersect_dicts(csd, model.state_dict())
            if csd == {}:
                csd = ckpt[key].float().state_dict()
                csd = add_module_prefix(csd)
                csd = intersect_dicts(csd, model.state_dict())
            model.load_state_dict(csd, strict=False)
            
        if ckpt['D_optimizer'] is not None:
            D_optimizer.load_state_dict(ckpt['D_optimizer'])
            lowest_loss = ckpt['lowest_loss']
        
        if ckpt['G_optimizer'] is not None:
            G_optimizer.load_state_dict(ckpt['G_optimizer'])
        
        start_epoch = ckpt['epoch'] + 1
        assert start_epoch > 0, f'{name} training to {epochs} epochs is finished, nothing to resume.'        
        del ckpt, csd
        print('resumed from epoch %d' % start_epoch)

    
    losses = {
        'gan_loss_fn' : nn.MSELoss(),
        'cycle_loss_fn' : nn.L1Loss(),
        'identity_loss_fn' : nn.L1Loss(),
    }

    # training/validation
    for epoch in range(start_epoch, end_epoch):
        print('epoch: %d/%d' % (epoch, end_epoch-1))
        t0 = time.time()
        
        # training
        epoch_loss = train_one_epoch(train_dataloader, G_XY, G_YX, D_X, D_Y, G_optimizer, D_optimizer, G_scaler, D_scaler, cuda, device, losses)
        t1 = time.time()
        print('loss=%.4f (took %.2f sec)' % (epoch_loss, t1-t0))
        
        # validation (just get sample images)
        val_one_epoch(val_dataloader, G_XY, device)
        
        weight_path = Path('weights')
        # # saving the best status into a weight file
        # Save model
        ckpt = {'epoch': epoch,
                'lowest_loss': lowest_loss,
                'G_XY': deepcopy(de_parallel(G_XY)).half(),
                'G_YX': deepcopy(de_parallel(G_YX)).half(),
                'D_X': deepcopy(de_parallel(D_X)).half(),
                'D_Y': deepcopy(de_parallel(D_Y)).half(),
                'D_optimizer': D_optimizer.state_dict(),
                'G_optimizer': G_optimizer.state_dict(),
                }
        
        if epoch_loss < lowest_loss:
            weight_file = weight_path /(name + '_best.pth')
            lowest_loss = epoch_loss
            torch.save(ckpt, weight_file)
            print('saved as a best model\n')

        weight_file = weight_path / (name + f'_{epoch}.pth')
        torch.save(ckpt, weight_file)
        print('saved as epoch\n')
        
        # wandb logging
        wandb.log({'train_loss': epoch_loss, })
        
def train_one_epoch(train_dataloader,  G_XY, G_YX, D_X, D_Y, G_optimizer, D_optimizer, G_scaler, D_scaler, cuda, device, losses, print_freq=10):
    G_XY.train()
    G_YX.train()
    D_X.train()
    D_Y.train()

    avg_G_loss = AverageMeter()
    avg_D_loss = AverageMeter()
    for i, (x, y, _, _) in enumerate(train_dataloader): # x: downsampled hr image, y: lr image
        x, y = x.to(device), y.to(device)
        
        # Update Discriminator (D) -----------------------------------------------
        shift_descriminator(D_X, D_Y, True)
        D_loss, fake_x, fake_y = get_d_loss(G_XY, G_YX, D_X, D_Y, losses, x, y, cuda)
        
        D_scaler.scale(D_loss).backward()
        D_scaler.step(D_optimizer)
        D_scaler.update()
        D_optimizer.zero_grad()
        
        # Update Generator (G)  ------------------------------------------------
        shift_descriminator(D_X, D_Y, False)
        G_loss = get_g_loss(G_XY, G_YX, D_X, D_Y, losses, x, y, fake_x, fake_y, cuda)
        
        G_scaler.scale(G_loss).backward()
        G_scaler.step(G_optimizer)
        G_scaler.update()
        G_optimizer.zero_grad()
        
        avg_G_loss.update(G_loss.item())
        avg_D_loss.update(D_loss.item())
        if i % print_freq == 0:
            print('\t iteration: %d/%d, G_loss=%.4f, D_loss=%.4f' % (i, len(train_dataloader)-1, avg_G_loss.value(), avg_D_loss.value()))
    total_loss = 0.7 * avg_G_loss.value() + 0.3 * avg_D_loss.value()
    return torch.tensor(total_loss)

def shift_descriminator(D_X, D_Y, mode = False):
    for param in D_X.parameters():
        param.requires_grad = mode
            
    for param in D_Y.parameters():
        param.requires_grad = mode

def get_d_loss(G_XY, G_YX, D_X, D_Y, losses, x, y, cuda):
    with amp.autocast(enabled=cuda):
        fake_y = G_XY(x)        # fake lr image (Nx1x512x512)
        real_y_score = D_Y(y)   # real lr score (Nx1x62x62)
        fake_y_score = D_Y(fake_y.detach())    # fake lr score (Nx1x62x62)
        real_y_loss = losses['gan_loss_fn'](real_y_score, torch.ones_like(real_y_score))
        fake_y_loss = losses['gan_loss_fn'](fake_y_score, torch.zeros_like(fake_y_score))
        D_y_loss = real_y_loss + fake_y_loss
        
        fake_x = G_YX(y)        # fake hr image (Nx1x512x512)
        real_x_score = D_X(x)
        fake_x_score = D_X(fake_x.detach())
        real_x_loss = losses['gan_loss_fn'](real_x_score, torch.ones_like(real_x_score))
        fake_x_loss = losses['gan_loss_fn'](fake_x_score, torch.zeros_like(fake_x_score))
        D_x_loss = real_x_loss + fake_x_loss
        
        D_loss = (D_x_loss + D_y_loss) / 2
        
    return D_loss, fake_x, fake_y

def get_g_loss(G_XY, G_YX, D_X, D_Y, losses, x, y, fake_x, fake_y, cuda):
    lambda_X, lambda_Y, lambda_I = 10,10,0.5
    with amp.autocast(enabled=cuda):
        # adversarial loss
        fake_x_score = D_X(fake_x)
        fake_y_score = D_Y(fake_y)
        fake_x_loss = losses['gan_loss_fn'](fake_x_score, torch.ones_like(fake_x_score))
        fake_y_loss = losses['gan_loss_fn'](fake_y_score, torch.ones_like(fake_y_score))
        # identity loss
        identity_x = G_YX(x)
        identity_y = G_XY(y)
        identity_x_loss = losses['identity_loss_fn'](identity_x, x) * lambda_X * lambda_I
        identity_y_loss = losses['identity_loss_fn'](identity_y, y) * lambda_Y * lambda_I
        # cycle loss
        cycle_x = G_YX(fake_y)
        cycle_y = G_XY(fake_x)
        cycle_x_loss = losses['cycle_loss_fn'](x, cycle_x) * lambda_X
        cycle_y_loss = losses['cycle_loss_fn'](y, cycle_y) * lambda_Y
        
        G_loss = fake_x_loss + fake_y_loss + identity_x_loss + identity_y_loss + cycle_x_loss + cycle_y_loss
    return G_loss

def val_one_epoch(val_dataloader, model, device):
    model.eval()
    for i, (ds_x, hr_x, filename) in enumerate(val_dataloader):
        ds_x, hr_x = ds_x.to(device), hr_x.to(device)
        fake_y = model(ds_x)
        
        if i == 0:
            for j in range(3):
                save_file = os.path.join('outputs', 'val_%d.png' % j)
                png_img = fake_y[j].mul(255.0).clamp(0,255).cpu().detach().numpy().squeeze(0).astype(np.uint8)
                cv2.imwrite(save_file, png_img)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=250, help='target epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    parser.add_argument('--name', default='jbnu_img_trans', help='name for the run')
    parser.add_argument('--save-dir', default='results', help='directory to save the results')
    parser.add_argument('--save-freq', type=int, default=1, help='save frequency')

    opt = parser.parse_args()

    train(opt)