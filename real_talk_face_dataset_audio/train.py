#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:09 2020

@author: pati
"""

# Needed libraries and packages
import os
import ast
import torch
import argparse
import matplotlib

import torch.nn as nn
import torch.optim as optim

from network.model import *
from network.blocks import *
from loss.loss_generator import *
from loss.loss_discriminator import *

from dataloader import CelebDataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils import  mount_server_disks, define_directories
from params import *

matplotlib.use('agg')
plt.ion()


# Construct and Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-link","--virtualLink", default=False, type=ast.literal_eval, 
    help="whether to mount virtual link to datasets in other computers")
ap.add_argument("-dir", "--datasetDir", default="../../../../../../media/gatv-server-i9/f7c33b0e-4235-4135-a649-cc5d2f4c1ce7/Preprocess/preprocessed",
	type=str, help="path to the input directory, where input files are stored")
ap.add_argument("-device", "--device", default="cuda:0", type=str,
	help="device on where script runs")
ap.add_argument("-reset", "--resetInfo", default=True, type=ast.literal_eval,
	help="whether to remove actual info saved") 

args = vars(ap.parse_args())


# Define dataset paths
path_list = [] # list with every path from different devices (Visiona 1, visiona 2 and server)


# Virtual link to datasets
if args["virtualLink"]:
    mount_dir_visiona1 = "/mnt/dataset_visiona1"
    mount_dir_visiona2 = "/mnt/dataset_visiona2"
    mount_server_disks(mount_dir_visiona1, mount_dir_visiona2)
    path_list.append("../../../../../.."+mount_dir_visiona1)
    path_list.append("../../../../../.."+mount_dir_visiona2)

path_list.append(args["datasetDir"])


# Define directories
logger = define_directories(args["resetInfo"], "model")


# Check if gpu available
device = args["device"]
if device == "cuda:0":
	gpu = torch.cuda.is_available()
	if gpu:
	    torch.cuda.set_device("cuda:0")


# Load dataset
dataset = CelebDataset(path_list=path_list, K=K, device=device, path_to_Wi=path_to_Wi)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Set model to train and send it to device
print('Initializating models...')
image_E = ImageEmbedder(256)
audio_E = AudioEmbedder(256)
G = Generator(256)
D = Discriminator(dataset.__len__(), path_to_Wi)

image_E.to(device)
audio_E.to(device)
G.to(device)
D.to(device)

optimizerG = optim.Adam(params = list(image_E.parameters()) + list(G.parameters()),
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
    image_E.apply(init_weights)
    audio_E.apply(init_weights)
    D.apply(init_weights)
    G.apply(init_weights)

    print('Initiating new audio model checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': batch_loss_G,
            'lossesD': batch_loss_D,
            'image_E_state_dict': image_E.state_dict(),
            'audio_E_state_dict': audio_E.state_dict(),
            'G_state_dict': G.state_dict(),
            'D_state_dict': D.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': optimizerG.state_dict(),
            'optimizerD': optimizerD.state_dict()
            }, path_to_chkpt)
    print('...Done')


# Loading from past checkpoint
print('Loading audio model checkpoint...')
checkpoint = torch.load(path_to_chkpt, map_location='cpu')
image_E.load_state_dict(GATV_checkpoint['image_E_state_dict'])
audio_E.load_state_dict(GATV_checkpoint['audio_E_state_dict'])
G.load_state_dict(checkpoint['G_state_dict'], strict=False)
D.load_state_dict(checkpoint['D_state_dict'])

epochCurrent = checkpoint['epoch']
batch_loss_G = checkpoint['lossesG']
batch_loss_D = checkpoint['lossesD']
num_vid = checkpoint['num_vid']
i_batch_current = checkpoint['i_batch'] +1
optimizerG.load_state_dict(checkpoint['optimizerG'])
optimizerD.load_state_dict(checkpoint['optimizerD'])  


# Set model to train mode
image_E.train()
audio_E.train()
G.train()
D.train()


# Training
lossD = []
lossG = []
image_E_weights = [] 
audio_E_weights = [] 
D_weights = [] 
G_weights = []

pbar = tqdm(dataLoader, leave=True, initial=0)

for epoch in range(epochCurrent, epochs):
    if epoch > epochCurrent:
        i_batch_current = 0
        pbar = tqdm(dataLoader, leave=True, initial=0)

    pbar.set_postfix(epoch=epoch)
    batch_local_loss_D = []
    batch_local_loss_G = []

    for i_batch, (f_lm, x, g_y, i, W_i, g_spect) in enumerate(pbar, start=0):
        f_lm = f_lm.to(device)
        x = x.to(device) # torch.Size([1, 3, 256, 256])
        g_y = g_y.to(device) # torch.Size([1, 3, 256, 256])
        g_spect = g_spect.to(device) # torch.Size([1, 3, 256, 256])
        W_i = W_i.squeeze(-1).transpose(0,1).to(device).requires_grad_()
        
        D.load_W_i(W_i)
        
        with torch.autograd.enable_grad():
            # Zero the parameter gradients
            optimizerG.zero_grad()
            optimizerD.zero_grad()

            # Forward
            # Calculate average encoding vector for video
            f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224

            #print("f_lm_compact ", np.shape(f_lm_compact)) # torch.Size([8, 2, 3, 256, 256])
            #print("f_lm_compact[:,0,:,:,:] ", np.shape(f_lm_compact[:,0,:,:,:])) # torch.Size([8, 3, 256, 256])
            #print("f_lm_compact[:,1,:,:,:] ", np.shape(f_lm_compact[:,1,:,:,:])) # torch.Size([8, 3, 256, 256])
            e_vectors = image_E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxK,512,1
            #print("e_vectors: ",np.shape(e_vectors)) # torch.Size([8, 512, 1])
            e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,K,512,1
            #print("e_vectors: ",np.shape(e_vectors)) # torch.Size([1, 8, 512, 1])
            images_vector = e_vectors.mean(dim=1)
            #print("images_vector: ",np.shape(images_vector)) # torch.Size([1, 512, 1])

            audio_vector = audio_E(g_spect) #B, 512, 1 ?? K sería 1 (no hay varios shots, es solo 1 imagen como lo que entra a G)
            #print("audio_vector: ",np.shape(audio_vector)) # torch.Size([1, 512, 1])


            # Concatenate embedders outputs
            e_hat = torch.cat((images_vector, audio_vector), 1)
            #print("e_hat: ",np.shape(e_hat)) # torch.Size([1, 1024, 1])

            # Train G and D
            x_hat = G(g_y, e_hat)
            #x_hat = G_GATV(g_y,images_vector)
            r_hat, D_hat_res_list = D(x_hat, g_y, i)
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

            r, D_res_list = D_GATV(x, g_y, i)
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
        batch_local_loss_D.append(lossD.item())
        batch_local_loss_G.append(lossG.item())
        batch_loss_G.append(lossG.item())
        batch_loss_D.append(lossD.item())

    # Federated learning
    lossD_avg = sum(batch_local_loss_D)/len(batch_local_loss_D)
    lossG_avg = sum(batch_local_loss_G)/len(batch_local_loss_G)

    # Output training stats      
    print('Saving latest audio model...')
    torch.save({
            'epoch': epoch+1,
            'lossesG': batch_loss_G,
            'lossesD': batch_loss_D,
            'image_E_state_dict': image_E.state_dict(),
            'audio_E_state_dict': audio_E.state_dict(),
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
    print('...Done saving latest audio model')

    if epoch%5 == 0:
        print('Saving latest audio model...')
        torch.save({
                'epoch': epoch+1,
                'lossesG': batch_loss_G,
                'lossesD': batchV_loss_D,
                'image_E_state_dict': image_E.state_dict(),
            	'audio_E_state_dict': audio_E.state_dict(),
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': i_batch,
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict()
                }, path_to_backup+"_"+str(epoch)+".tar")
        out = (x_hat[0]*255).transpose(0,2)
        out = out.type(torch.uint8).to("cpu").numpy()
        plt.imsave(training_image_path+str(epoch)+".png", out)
        print('...Done saving latest audio model')   
