#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:18 2020

@author: pati
"""
import os
import torch
import argparse
import ast
import shutil

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils import mountDisksGatv, define_directories

from torch.utils.data import DataLoader
from dataset_class import VidDataSet
from params.params import *

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')
plt.ion()


# Construct and Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-link","--virtualLink", default = False, type = ast.literal_eval, 
    help = "whether to mount virtual link to datasets in other computers")
ap.add_argument("-dir", "--datasetDir", default = "../../../../../../media/gatvprojects/21C6DF8B198B99B31/Preprocess/preprocessed", type = str, 
    help = 'path to the input directory, where input files are stored.')
ap.add_argument("-reset", "--resetInfo", default = False, type = ast.literal_eval,
    help="whether to remove actual info saved")
args = vars(ap.parse_args())

# Define dataset paths
path_list = [] # list with every path from different devices (Visiona 1, visiona 2 and server)
path_list.append(args["datasetDir"])

# Virtual link to datasets
if args["virtualLink"]:
    mount_dir_famous = "/mnt/dataset_dir_famous"
    mount_dir_tfg_tfm= "/mnt/dataset_dir_tfg_tfm"
    mountDisksGatv(mount_dir_famous, mount_dir_tfg_tfm)
    path_list.append("../../../../../.."+mount_dir_famous)
    path_list.append("../../../../../.."+mount_dir_tfg_tfm)

# Define directories
logger = define_directories(args["resetInfo"], "gatv")

# Check if gpu available
gpu = torch.cuda.is_available()
if gpu:
    torch.cuda.set_device("cuda:0")
    
device = 'cuda:0' if gpu else 'cpu'

# Load dataset
dataset = VidDataSet(K=K, path_list=path_list, device=device, path_to_Wi=gatv_path_to_Wi)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

# Set model to train and send it to device
print('Initializating models...')
E_GATV = Embedder(256)
G_GATV = Generator(256)
D_GATV= Discriminator(dataset.__len__(), gatv_path_to_Wi)

E_GATV.to(device)
G_GATV.to(device)
D_GATV.to(device)

GATV_optimizerG = optim.Adam(params = list(E_GATV.parameters()) + list(G_GATV.parameters()),
                    lr=5e-5,
                    amsgrad=False)
GATV_optimizerD = optim.Adam(params = D_GATV.parameters(),
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
batch_GATV_loss_G = []
batch_GATV_loss_D = []

# Initiate gatv checkpoint if inexistant
if not os.path.isfile(gatv_path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
    E_GATV.apply(init_weights)
    D_GATV.apply(init_weights)
    G_GATV.apply(init_weights)

    print('Initiating new GATV checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': batch_GATV_loss_G,
            'lossesD': batch_GATV_loss_D,
            'E_state_dict': E_GATV.state_dict(),
            'G_state_dict': G_GATV.state_dict(),
            'D_state_dict': D_GATV.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': GATV_optimizerG.state_dict(),
            'optimizerD': GATV_optimizerD.state_dict()
            }, gatv_path_to_chkpt)
    print('...Done')

# Loading from past gatv checkpoint
print('Loading GATV checkpoint...')
GATV_checkpoint = torch.load(gatv_path_to_chkpt, map_location='cpu')
E_GATV.load_state_dict(GATV_checkpoint['E_state_dict'])
G_GATV.load_state_dict(GATV_checkpoint['G_state_dict'], strict=False)

GATV_checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32)
D_GATV.load_state_dict(GATV_checkpoint['D_state_dict'])

epochCurrent = GATV_checkpoint['epoch']
batch_GATV_loss_G = GATV_checkpoint['lossesG']
batch_GATV_loss_D = GATV_checkpoint['lossesD']
num_vid = GATV_checkpoint['num_vid']
i_batch_current = GATV_checkpoint['i_batch'] +1
GATV_optimizerG.load_state_dict(GATV_checkpoint['optimizerG'])
GATV_optimizerD.load_state_dict(GATV_checkpoint['optimizerD'])  

# Set model to train mode
E_GATV.train()
D_GATV.train()
G_GATV.train()


# Training
GATV_lossD = []
GATV_lossG = []
GATV_E_weights = [] 
GATV_D_weights = [] 
GATV_G_weights = []

pbar = tqdm(dataLoader, leave=True, initial=0)

for epoch in range(epochCurrent, epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(dataLoader, leave=True, initial=0)

    pbar.set_postfix(epoch=epoch)
    batch_loss_D = []
    batch_loss_G = []

    for i_batch, (f_lm, x, g_y, i, W_i) in enumerate(pbar, start=0):
        f_lm = f_lm.to(device)
        x = x.to(device)
        g_y = g_y.to(device)
        W_i = W_i.squeeze(-1).transpose(0,1).to(device).requires_grad_()
        
        D_GATV.load_W_i(W_i)
        
        with torch.autograd.enable_grad():
            #zero the parameter gradients
            GATV_optimizerG.zero_grad()
            GATV_optimizerD.zero_grad()

            #forward
            # Calculate average encoding vector for video
            f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224

            e_vectors = E_GATV(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxK,512,1
            e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,K,512,1
            e_hat = e_vectors.mean(dim=1)

            #train G and D
            x_hat = G_GATV(g_y, e_hat)
            r_hat, D_hat_res_list = D_GATV(x_hat, g_y, i)
            with torch.no_grad():
                r, D_res_list = D_GATV(x, g_y, i)

            lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, D_GATV.W_i, i)
     
            lossG.backward(retain_graph=False)
            GATV_optimizerG.step()
        
        with torch.autograd.enable_grad():
            GATV_optimizerG.zero_grad()
            GATV_optimizerD.zero_grad()
            x_hat.detach_().requires_grad_()
            r_hat, D_hat_res_list = D_GATV(x_hat, g_y, i)
            lossDfake = criterionDfake(r_hat)

            r, D_res_list = D_GATV(x, g_y, i)
            lossDreal = criterionDreal(r)
            
            lossD = lossDfake + lossDreal
            lossD.backward(retain_graph=False)
            GATV_optimizerD.step()
            
            GATV_optimizerD.zero_grad()
            r_hat, D_hat_res_list = D_GATV(x_hat, g_y, i)
            lossDfake = criterionDfake(r_hat)

            r, D_res_list = D_GATV(x, g_y, i)
            lossDreal = criterionDreal(r)
            
            lossD = lossDfake + lossDreal
            lossD.backward(retain_graph=False)
            GATV_optimizerD.step()

        # Save discriminator weights
        for enum, idx in enumerate(i):
            torch.save({'W_i': D_GATV.W_i[:,enum].unsqueeze(-1)}, gatv_path_to_Wi+'/W_'+str(idx.item())+'/W_'+str(idx.item())+'.tar')
        
        # Save loss from batches each epoch
        batch_loss_D.append(lossD.item())
        batch_loss_G.append(lossG.item())
        batch_GATV_loss_G.append(lossG.item())
        batch_GATV_loss_D.append(lossD.item())

    # Federated learning
    lossD_avg = sum(batch_loss_D)/len(batch_loss_D)
    lossG_avg = sum(batch_loss_G)/len(batch_loss_G)

    torch.save({
        'epoch': epoch+1, 
        'E_state_dict': E_GATV.state_dict(), 
        'G_state_dict': G_GATV.state_dict(), 
        'D_state_dict': D_GATV.state_dict(), 
        'num_vid': dataset.__len__(),
        'i_batch': i_batch,
        'optimizerG': GATV_optimizerG.state_dict(), 
        'optimizerD': GATV_optimizerD.state_dict(), 
        'lossesG': lossG_avg, 
        'lossesD': lossD_avg}, 
        'local_model_'+str(epoch)+'.pth')

    data = open('local_model_'+str(epoch)+'.pth', 'rb')
    files = {'file': data}

    r = requests.post("http://192.168.0.47:5000/average_models/", files=files)


    # Output training stats      
    print('Saving latest GATV model...')
    torch.save({
            'epoch': epoch+1,
            'lossesG': batch_GATV_loss_G,
            'lossesD': batch_GATV_loss_D,
            'E_state_dict': E_GATV.state_dict(),
            'G_state_dict': G_GATV.state_dict(),
            'D_state_dict': D_GATV.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': GATV_optimizerG.state_dict(),
            'optimizerD': GATV_optimizerD.state_dict()
            }, gatv_path_to_chkpt)
    out = (x_hat[0]*255).transpose(0,2)
    out = out.type(torch.uint8).to("cpu").numpy()
    plt.imsave(gatv_training_image_path+str(epoch)+".png", out)
    print('...Done saving latest GATV model')

    if epoch%5 == 0:
        print('Saving latest GATV model...')
        torch.save({
                'epoch': epoch+1,
                'lossesG': batch_GATV_loss_G,
                'lossesD': batch_GATV_loss_D,
                'E_state_dict': E_GATV.state_dict(),
                'G_state_dict': G_GATV.state_dict(),
                'D_state_dict': D_GATV.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': i_batch,
                'optimizerG': GATV_optimizerG.state_dict(),
                'optimizerD': GATV_optimizerD.state_dict()
                }, gatv_path_to_backup+"_"+str(epoch)+".tar")
        print('...Done saving latest GATV model')   