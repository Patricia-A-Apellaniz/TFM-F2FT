#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:18 2020

@author: pati
"""
import os
import ast
import copy
import torch
import shutil
import argparse
import matplotlib

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from dataloader import VidDataSet
from utils import mountDisksServer, define_directories
from tensorboardX import SummaryWriter

from torch.utils.data import DataLoader
from params import *

from network.model import *
from network.blocks import *
from loss.loss_generator import *
from loss.loss_discriminator import *

from matplotlib import pyplot as plt
matplotlib.use('agg')
plt.ion()


# Construct and Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-link","--virtualLink", default = False, type = ast.literal_eval, 
	help = "whether to mount virtual link to datasets in other computers")
ap.add_argument("-dir", "--datasetDir", default = "../../../../../../media/gatv-server-i9/f7c33b0e-4235-4135-a649-cc5d2f4c1ce7/Preprocess/preprocessed", type = str, 
	help = 'path to the input directory, where input files are stored.')
ap.add_argument("-reset", "--resetInfo", default = False, type = ast.literal_eval,
    help="whether to remove actual info saved") 
args = vars(ap.parse_args())

# Define dataset paths
path_list = [] # list with every path from different devices (Visiona 1, visiona 2 and server)
path_list.append(args["datasetDir"])

# Virtual link to datasets
if args["virtualLink"]:
    mount_dir_visiona1 = "/mnt/dataset_visiona1"
    mount_dir_visiona2 = "/mnt/dataset_visiona2"
    mountDisksServer(mount_dir_visiona1, mount_dir_visiona2)
    path_list.append("../../../../../.."+mount_dir_visiona1)
    path_list.append("../../../../../.."+mount_dir_visiona2)

# Define directories
logger = define_directories(args["resetInfo"], "model")

# Check if gpu available
gpu = torch.cuda.is_available()
if gpu:
    torch.cuda.set_device("cuda:0") 
device = 'cuda:0' if gpu else 'cpu'

# Load dataset
dataset = VidDataSet(K=K, path_list=path_list, device=device, path_to_Wi=path_to_Wi)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

# Set model to train and send it to device
print('Initializating models...')
E = Embedder(256)
G = Generator(256)
D = Discriminator(dataset.__len__(), path_to_Wi)

E.to(device)
D.to(device)
G.to(device)

optimizerG = optim.Adam(params = list(E.parameters()) + list(G.parameters()),
                    lr=5e-5,
                    amsgrad=False)
optimizerD = optim.Adam(params = D.parameters(),
                    lr=2e-4,
                    amsgrad=False)

criterionG = LossG(VGGFace_body_path='Pytorch_VGGFACE_IR.py',
	VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()		

# Training initialization
print('Initializating training...')
epochCurrent = epoch = i_batch = 0
i_batch_current = 0
batch_loss_G = []
batch_loss_D = []

# Initiate checkpoint if inexistant
if not os.path.isfile(path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
    E.apply(init_weights)
    D.apply(init_weights)
    G.apply(init_weights)

    print('Initiating new checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': batch_loss_G,
            'lossesD': batch_loss_D,
            'E_state_dict': E.state_dict(),
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
            }, path_to_chkpt)
    print('...Done')

# Loading from past checkpoint
print('Loading checkpoint...')
checkpoint = torch.load(path_to_chkpt, map_location='cpu')
E.load_state_dict(checkpoint['E_state_dict'])
G.load_state_dict(checkpoint['G_state_dict'], strict=False)

checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32)
D.load_state_dict(checkpoint['D_state_dict'])

epochCurrent = checkpoint['epoch']
batch_loss_G = checkpoint['lossesG']
batch_loss_D = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] +1
optimizerG.load_state_dict(checkpoint['optimizerG'])
optimizerD.load_state_dict(checkpoint['optimizerD'])

# Set model to train mode
G.train()
E.train()
D.train()

# Training
pbar = tqdm(dataLoader, leave=True, initial=0)
for epoch in range(epochCurrent, epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(dataLoader, leave=True, initial=0)

    pbar.set_postfix(epoch=epoch)
    for i_batch, (f_lm, x, g_y, i, W_i) in enumerate(pbar, start=i_batch_current):
        print("i_batch: ", i_batch)
        f_lm = f_lm.to(device)
        x = x.to(device)
        g_y = g_y.to(device) # 2, 3, 256, 256
        W_i = W_i.squeeze(-1).transpose(0,1).to(device).requires_grad_() # 2, 512, 1 --> 512, 2
        
        D.load_W_i(W_i)
        with torch.autograd.enable_grad():
            # Zero the parameter gradients
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            # Forward
            # Calculate average encoding vector for video
            f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224

            e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxK,512,1
            e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,K,512,1
            e_hat = e_vectors.mean(dim=1) # 2,512,1

            #train G and D
            x_hat = G(g_y, e_hat) # 2, 3, 256, 256       
            r_hat, D_hat_res_list = D(x_hat, g_y, i) # 2x1x1 , 7

            with torch.no_grad():
                r, D_res_list = D(x, g_y, i)

            lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, D.W_i, i)
            lossG.backward(retain_graph=False)
            optimizerG.step()
    
        with torch.autograd.enable_grad():
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            x_hat.detach_().requires_grad_()
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            lossDfake = criterionDfake(r_hat)

            r, D_res_list = D(x, g_y, i)
            lossDreal = criterionDreal(r)

            lossD = lossDfake + lossDreal
            lossD.backward(retain_graph=False)
            optimizerD.step()

            optimizerD.zero_grad()
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
            lossDfake = criterionDfake(r_hat)

            r, D_res_list = D(x, g_y, i)
            lossDreal = criterionDreal(r)

            lossD = lossDfake + lossDreal
            lossD.backward(retain_graph=False)
            optimizerD.step()

        # Save discriminator weights
        for enum, idx in enumerate(i):
            torch.save({'W_i': D.W_i[:,enum].unsqueeze(-1)}, path_to_Wi+'/W_'+str(idx.item())+'/W_'+str(idx.item())+'.tar')

        # Save loss from batches each epoch
        print("LossG: ",lossG)
        print("LossG: ",lossG.item())
        print("LossD: ",lossD)
        print("LossD: ",lossD.item())
        batch_loss_G.append(lossG.item())
        batch_loss_D.append(lossD.item())

    # Output training stats      
    print('Saving latest model...')
    torch.save({
        'epoch': epoch+1,
        'lossesG': batch_loss_G,
        'lossesD': batch_loss_D,
        'E_state_dict': E.state_dict(),
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'num_vid': dataset.__len__(),
        'i_batch': i_batch,
        'optimizerG': optimizerG.state_dict(),
        'optimizerD': optimizerD.state_dict()
    }, path_to_chkpt)
    out = (x_hat[0]*255).transpose(0,2)
    out = out.type(torch.uint8).to("cpu").numpy()
    plt.imsave(training_image_path+str(epoch)+".png", out)
    print('...Done saving latest model')

    if epoch%5 == 0:
        print('Saving latest model...')
        torch.save({
            'epoch': epoch+1,
            'lossesG': batch_loss_G,
            'lossesD': batch_loss_D,
            'E_state_dict': E.state_dict(),
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
        }, path_to_backup+"_"+str(epoch)+".tar")
        print('...Done saving latest  model')



