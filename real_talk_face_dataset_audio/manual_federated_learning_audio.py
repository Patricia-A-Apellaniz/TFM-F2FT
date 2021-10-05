#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 20:18 2020

@author: pati
"""

# Needed libraries and packages
import os
import ast
import copy
import torch
import shutil
import random
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim

from params import *
from network.model import *
from network.blocks import *
from loss.loss_generator import *
from loss.loss_discriminator import *

from pathlib import Path
from dataloader import CelebDataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils import define_global_directories
from tensorboardX import SummaryWriter



# Returns the average of the weights.
def average_weights(w, pond):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key]*pond[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]*pond[i]
        #w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


# This function computes average weights
def average_models(epoch):
    pond = [0.9, 0.1]
    print("Computing average weights...")
    local_G = [G_server.state_dict(), G_gatv.state_dict()]
    local_D = [D_server.state_dict(), D_gatv.state_dict()]
    local_E_image = [E_image_server.state_dict(), E_image_gatv.state_dict()]
    local_E_audio = [E_audio_server.state_dict(), E_audio_gatv.state_dict()]

    local_loss_G = [batch_server_loss_G, batch_gatv_loss_G]
    local_loss_D = [batch_server_loss_D, batch_gatv_loss_D]

    # Update global weights
    E_global_state_image = average_weights(local_E_image, pond)
    E_global_state_audio = average_weights(local_E_audio, pond)
    D_global_state = average_weights(local_D, pond)
    G_global_state = average_weights(local_G, pond)

    E_image_global.load_state_dict(E_global_state_image)
    E_audio_global.load_state_dict(E_global_state_audio)
    G_global.load_state_dict(G_global_state, strict=False)
    D_global.W_i =  nn.Parameter(torch.rand(512, 32))
    D_global.load_state_dict(D_global_state)

    global_lossesG = sum(local_loss_G)/len(local_loss_G)
    global_lossesD = sum(local_loss_D)/len(local_loss_D)

    # Generate image
    #idx = random.randint(0, num_vid_server-1)
    #f_lm, x, g_y, i, W_i = dataset.__getitem__(idx)
    for i_batch, (f_lm, x, g_y, i, W_i_images, W_i_audio, g_spect) in enumerate(dataloader, start=0):
        if i_batch>0:
            break
        f_lm = f_lm.to(device)
        x = x.to(device)
        g_y = g_y.to(device)
        g_spect = g_spect.to(device)
        W_i_images = W_i_images.squeeze(-1).transpose(0,1).to(device).requires_grad_()
        W_i_audio = W_i_audio.squeeze(-1).transpose(0,1).to(device).requires_grad_()

        D_global.load_W_i(W_i)
        with torch.autograd.enable_grad():
            global_optimizerG.zero_grad()
            global_optimizerD.zero_grad()
            f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) 

            e_vectors = E_image_global(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) 
            e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) 
            e_hat = e_vectors.mean(dim=1) 
            audio_vector = E_audio_global(g_spect)

            # Concatenate embedders outputs
            e_hat = torch.cat((e_vectors, audio_vector), 1)

            x_hat = G_global(g_y, e_hat)    
            x_hat.detach_().requires_grad_()

    torch.save({
            'lossesG': global_lossesG,
            'lossesD': global_lossesD,
            'image_E_state_dict': E_image_global.state_dict(),
            'audio_E_state_dict': E_audio_global.state_dict(),
            'G_state_dict': G_global.state_dict(),
            'D_state_dict': D_global.state_dict(),
            'optimizerG': global_optimizerG.state_dict(),
            'optimizerD': global_optimizerD.state_dict()
            }, global_path_to_backup+"_"+str(epoch)+".tar")
    print('...Done saving latest average model')
    out = (x_hat[0]*255).transpose(0,2)
    out = out.type(torch.uint8).to("cpu").numpy()
    plt.imsave(global_training_image_path+str(epoch)+".png", out)

# Create global folder
define_global_directories()

# Check if gpu available
gpu = torch.cuda.is_available()
if gpu:
    torch.cuda.set_device("cuda:0") 
device = 'cuda:0' if gpu else 'cpu'

# Load dataset
path_list = [] # list with every path from different devices (Visiona 1, visiona 2 and server)
path_list.append("../../../../../../media/gatv-server-i9/f7c33b0e-4235-4135-a649-cc5d2f4c1ce7/Preprocess/preprocessed")
dataset = CelebDataset(K=K, path_list=path_list, device=device, path_to_Wi=server_path_to_Wi)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

# Load server checkpoint
print('Server model...')
server_checkpoint = torch.load(server_path_to_chkpt, map_location='cpu')
num_vid_server = server_checkpoint['num_vid']
print("Number videos server: ", num_vid_server)

E_image_server = ImageEmbedder(256)
E_audio_server = AudioEmbedder(256)
G_server = Generator(256)
D_server = Discriminator(num_vid_server, server_path_to_Wi)
E_image_server.to(device)
E_audio_server.to(device)
G_server.to(device)
D_server.to(device)

# Load GATV checkpoint
print('GATV  model...')
gatv_checkpoint = torch.load(gatv_path_to_chkpt, map_location='cpu')
num_vid_gatv = gatv_checkpoint['num_vid']
print("Number videos gatv: ", num_vid_gatv)

E_image_gatv = ImageEmbedder(256)
E_audio_gatv = AudioEmbedder(256)
G_gatv = Generator(256)
D_gatv = Discriminator(num_vid_gatv, gatv_path_to_Wi)
E_image_gatv.to(device)
E_audio_gatv.to(device)
G_gatv.to(device)
D_gatv.to(device)

# Creating global model and loading from past global checkpoint
print("Initializing global checkpoint...")
E_image_global = ImageEmbedder(256)
E_audio_global = AudioEmbedder(256)
G_global = Generator(256)
D_global = Discriminator(num_vid_server, global_path_to_Wi)
E_image_global.to(device)
E_audio_global.to(device)
D_global.to(device)
G_global.to(device)

global_optimizerG = optim.Adam(params = list(E_global.parameters()) + list(G_global.parameters()),
                    lr=5e-5,
                    amsgrad=False)
global_optimizerD = optim.Adam(params = D_global.parameters(),
                    lr=2e-4,
                    amsgrad=False)

# Initiate global checkpoint if inexistant
if os.path.isfile(global_path_to_chkpt):
	shutil.rmtree(global_path_to_chkpt, ignore_errors=True)

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
E_image_global.apply(init_weights)
E_audio_global.apply(init_weights)
D_global.apply(init_weights)
G_global.apply(init_weights)

global_epoch = 0
global_lossesG = 0
global_lossesD = 0
print('Initiating new global checkpoint...')
torch.save({
        'lossesG': global_lossesG,
        'lossesD': global_lossesD,
        'image_E_state_dict': E_image_global.state_dict(),
        'audio_E_state_dict': E_audio_global.state_dict(),
        'G_state_dict': G_global.state_dict(),
        'D_state_dict': D_global.state_dict(),
        'optimizerG': global_optimizerG.state_dict(),
        'optimizerD': global_optimizerD.state_dict()
        }, global_path_to_backup+'_'+str(global_epoch)+'.tar')
print('...Done')

# Loading from past global checkpoint
print('Loading global checkpoint...')
global_checkpoint = torch.load(global_path_to_backup+'_'+str(global_epoch)+'.tar', map_location='cpu')
E_image_global.load_state_dict(global_checkpoint['image_E_state_dict'])
E_audio_global.load_state_dict(global_checkpoint['audio_E_state_dict'])
G_global.load_state_dict(global_checkpoint['G_state_dict'], strict=False)
global_checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32)
D_global.load_state_dict(global_checkpoint['D_state_dict'], strict=False)
global_lossesG = global_checkpoint['lossesG']
global_lossesD = global_checkpoint['lossesD']
global_optimizerG.load_state_dict(global_checkpoint['optimizerG'])
global_optimizerD.load_state_dict(global_checkpoint['optimizerD'])


# Read each server backup file
server_path_to_weights = str(server_path_to_chkpt).split("/")[0]+"/"+str(server_path_to_chkpt).split("/")[1]+"/"
server_path = Path(server_path_to_weights)
for server_backup in server_path.iterdir():
    # Take epoch and search for same epoch gatv backup
    file_name = str(server_backup).split(".")[0]
    model_epoch = file_name.split("_")[-1]

    gatv_backup = gatv_path_to_backup+"_"+str(model_epoch)+".tar"
    if os.path.exists(gatv_backup) and model_epoch != "0":
        print('Loading server '+str(model_epoch)+' checkpoint...')
        server_backup_checkpoint = torch.load(server_backup, map_location='cpu')
        E_image_server.load_state_dict(server_backup_checkpoint['image_E_state_dict'])
        E_audio_server.load_state_dict(server_backup_checkpoint['audio_E_state_dict'])
        G_server.load_state_dict(server_backup_checkpoint['G_state_dict'], strict=False)
        server_backup_checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32)
        D_server.load_state_dict(server_backup_checkpoint['D_state_dict'])
        batch_server_loss_G = server_backup_checkpoint['lossesG']
        batch_server_loss_G = batch_server_loss_G[(len(batch_server_loss_G)-num_vid_server-1):]
        batch_server_loss_G = sum(batch_server_loss_G)/len(batch_server_loss_G)
        batch_server_loss_D = server_backup_checkpoint['lossesD']
        batch_server_loss_D = batch_server_loss_D[(len(batch_server_loss_D)-num_vid_server-1):]
        batch_server_loss_D = sum(batch_server_loss_D)/len(batch_server_loss_D)

        print('Loading GATV '+str(model_epoch)+' checkpoint...')
        gatv_backup_checkpoint = torch.load(gatv_backup, map_location='cpu')
        E_image_gatv.load_state_dict(gatv_backup_checkpoint['image_E_state_dict'])
        E_audio_gatv.load_state_dict(gatv_backup_checkpoint['audio_E_state_dict'])
        G_gatv.load_state_dict(gatv_backup_checkpoint['G_state_dict'], strict=False)
        gatv_backup_checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32)
        D_gatv.load_state_dict(gatv_backup_checkpoint['D_state_dict'])
        batch_gatv_loss_G = gatv_backup_checkpoint['lossesG']
        batch_gatv_loss_G = batch_gatv_loss_G[(len(batch_gatv_loss_G)-num_vid_gatv-1):]
        batch_gatv_loss_G = sum(batch_gatv_loss_G)/len(batch_gatv_loss_G)
        batch_gatv_loss_D = gatv_backup_checkpoint['lossesD']
        batch_gatv_loss_D = batch_gatv_loss_D[(len(batch_gatv_loss_D)-num_vid_gatv-1):]
        batch_gatv_loss_D = sum(batch_gatv_loss_D)/len(batch_gatv_loss_D)

        # Compute average weights
        average_models(model_epoch)

