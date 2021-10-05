#### THIS IS SUPPOSED TO BE RUN FROM SERVER WHERE GLOBAL MODEL IS STORED####
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
import copy

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils import mountDisksServer, define_directories, define_global_directories

from torch.utils.data import DataLoader
from dataloader import VidDataSet
from params import *

from loss.loss_discriminator import *
from loss.loss_generator import *
from network.blocks import *
from network.model import *

from flask import Flask, request, Response
from tensorboardX import SummaryWriter
import jsonpickle
import threading
from werkzeug.utils import secure_filename

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')
plt.ion()


app = Flask(__name__)

# Shared files configuration
UPLOAD_FOLDER = 'gatv_weights_uploaded/'
if not os.path.exists(UPLOAD_FOLDER):
	os.mkdir(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = set(['pth'])

def allowed_file(filename):
    return '.' in filename and \
       filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# Returns the average of the weights.
def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


GATV_E_weights = []
GATV_G_weights = []
GATV_D_weights = []
GATV_lossG = []
GATV_lossD = []

# This function computes average weigths
epoch = 0
def average_models():
    if len(GATV_E_weights)>0 and len(server_E_weights)>0 and len(GATV_D_weights)>0 and len(server_D_weights)>0 and len(GATV_G_weights)>0 and len(server_G_weights)>0:
        print("Computing average weigths...")
        local_weights_G = [server_G_weights[0], GATV_G_weights[0]]
        local_weights_D = [server_D_weights[0], GATV_D_weights[0]]
        local_weights_E = [server_E_weights[0], GATV_E_weights[0]]

        local_loss_G = [server_lossG[0], GATV_lossG[0]]
        local_loss_D = [server_lossD[0], GATV_lossD[0]]

        # Remove item
        server_G_weights.pop(0)
        server_D_weights.pop(0)
        server_E_weights.pop(0)
       	server_lossG.pop(0)
        server_lossD.pop(0)
        GATV_G_weights.pop(0)
        GATV_D_weights.pop(0)
        GATV_E_weights.pop(0)
        GATV_lossG.pop(0)
        GATV_lossD.pop(0)

        # Update global weights
        E_global_state = average_weights(local_weights_E)
        D_global_state = average_weights(local_weights_D)
        G_global_state = average_weights(local_weights_G)

        E_global.load_state_dict(E_global_state)
        G_global.load_state_dict(G_global_state, strict=False)
        D_global.W_i =  nn.Parameter(torch.rand(512, 32))
        D_global.load_state_dict(D_global_state)

        global_lossesG = sum(local_loss_G) / len(local_loss_G)
        global_lossesD = sum(local_loss_D) / len(local_loss_D)

        for i_batch, (f_lm, x, g_y, i, W_i) in enumerate(dataloader, start=0):
            if i_batch>0:
                break
            f_lm = f_lm.to(device)
            x = x.to(device)
            g_y = g_y.to(device)
            W_i = W_i.squeeze(-1).transpose(0,1).to(device).requires_grad_()

            D_global.load_W_i(W_i)
            with torch.autograd.enable_grad():
                global_optimizerG.zero_grad()
                global_optimizerD.zero_grad()
                f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) 

                e_vectors = E_global(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) 
                e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) 
                e_hat = e_vectors.mean(dim=1) 

                x_hat = G_global(g_y, e_hat)    
                x_hat.detach_().requires_grad_()

        print('Saving latest average model...')
        torch.save({
                'lossesG': global_lossesG,
                'lossesD': global_lossesD,
                'E_state_dict': E_global.state_dict(),
                'G_state_dict': G_global.state_dict(),
                'D_state_dict': D_global.state_dict(),
                'optimizerG': global_optimizerG.state_dict(),
                'optimizerD': global_optimizerD.state_dict()
                }, global_path_to_backup+"_"+str(epoch)+".tar")
        print('...Done saving latest server model')
        out = (x_hat[0]*255).transpose(0,2)
        out = out.type(torch.uint8).to("cpu").numpy()
        plt.imsave(global_training_image_path+str(epoch)+".png", out)

        ###### Use model to generate image
        

# Construct and Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-link","--virtualLink", default = False, type = ast.literal_eval, 
	help = "whether to mount virtual link to datasets in other computers")
ap.add_argument("-dir", "--datasetDir", default = "../../../../../../media/gatv-server-i9/f7c33b0e-4235-4135-a649-cc5d2f4c1ce7/Preprocess/preprocessed", type = str, 
	help = 'path to the input directory, where input files are stored.')
ap.add_argument("-reset", "--resetInfo", default = True, type = ast.literal_eval,
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
logger = define_directories(args["resetInfo"], "server")
define_global_directories()

# Check if gpu available
gpu = torch.cuda.is_available()
if gpu:
    torch.cuda.set_device("cuda:0")
    
device = 'cuda:0' if gpu else 'cpu'

# Load dataset
dataset = VidDataSet(K=K, path_list=path_list, device=device, path_to_Wi=server_path_to_Wi)
dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) 

# Set model to train and send it to device
print('Initializating models...')
E_global = Embedder(256)
G_global = Generator(256)
D_global = Discriminator(dataset.__len__(), server_path_to_Wi)

E_server = Embedder(256)
G_server = Generator(256)
D_server= Discriminator(dataset.__len__(), server_path_to_Wi)

E_global.to(device)
D_global.to(device)
G_global.to(device)

E_server.to(device)
D_server.to(device)
G_server.to(device)

server_optimizerG = optim.Adam(params = list(E_server.parameters()) + list(G_server.parameters()),
                    lr=5e-5,
                    amsgrad=False)
server_optimizerD = optim.Adam(params = D_server.parameters(),
                    lr=2e-4,
                    amsgrad=False)

global_optimizerG = optim.Adam(params = list(E_global.parameters()) + list(G_global.parameters()),
                    lr=5e-5,
                    amsgrad=False)
global_optimizerD = optim.Adam(params = D_global.parameters(),
                    lr=2e-4,
                    amsgrad=False)

criterionG = LossG(VGGFace_body_path='Pytorch_VGGFACE_IR.py',
	VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)
criterionDreal = LossDSCreal()
criterionDfake = LossDSCfake()		

# Copy global weights
E_global_state = E_global.state_dict()
G_global_state = G_global.state_dict()
D_global_state = D_global.state_dict()
opG_global_state = global_optimizerG.state_dict()
opD_global_state = global_optimizerD.state_dict()

# Training initialization
print('Initializating training...')
epochCurrent = i_batch = 0
i_batch_current = 0
batch_server_loss_G = []
batch_server_loss_D = []
global_lossesG = 0
global_lossesD = 0

# Initiate server checkpoint if inexistant
if not os.path.isfile(server_path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
    E_server.apply(init_weights)
    D_server.apply(init_weights)
    G_server.apply(init_weights)

    print('Initiating new server checkpoint...')
    torch.save({
            'epoch': epoch,
            'lossesG': batch_server_loss_G,
            'lossesD': batch_server_loss_D,
            'E_state_dict': E_server.state_dict(),
            'G_state_dict': G_server.state_dict(),
            'D_state_dict': D_server.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': server_optimizerG.state_dict(),
            'optimizerD': server_optimizerD.state_dict()
            }, server_path_to_chkpt)
    print('...Done')

# Loading from past server checkpoint
print('Loading server checkpoint...')
server_checkpoint = torch.load(server_path_to_chkpt, map_location='cpu')
E_server.load_state_dict(server_checkpoint['E_state_dict'])
G_server.load_state_dict(server_checkpoint['G_state_dict'], strict=False)

server_checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32)
D_server.load_state_dict(server_checkpoint['D_state_dict'])

epochCurrent = server_checkpoint['epoch']
batch_server_loss_G = server_checkpoint['lossesG']
batch_server_loss_D = server_checkpoint['lossesD']
num_vid = server_checkpoint['num_vid']
i_batch_current = server_checkpoint['i_batch'] +1
server_optimizerG.load_state_dict(server_checkpoint['optimizerG'])
server_optimizerD.load_state_dict(server_checkpoint['optimizerD'])

# Initiate global checkpoint if inexistant
if not os.path.isfile(global_path_to_chkpt):
    def init_weights(m):
        if type(m) == nn.Conv2d:
            torch.nn.init.xavier_uniform(m.weight)
    E_global.apply(init_weights)
    D_global.apply(init_weights)
    G_global.apply(init_weights)

    print('Initiating new global checkpoint...')
    torch.save({
            'lossesG': global_lossesG,
            'lossesD': global_lossesD,
            'E_state_dict': E_global.state_dict(),
            'G_state_dict': G_global.state_dict(),
            'D_state_dict': D_global.state_dict(),
            'optimizerG': global_optimizerG.state_dict(),
            'optimizerD': global_optimizerD.state_dict()
            }, global_path_to_chkpt+'_'+str(epoch)+'.tar')
    print('...Done')

# Loading from past global checkpoint
print('Loading global checkpoint...')
global_checkpoint = torch.load(global_path_to_chkpt+'_'+str(epoch)+'.tar', map_location='cpu')
E_global.load_state_dict(global_checkpoint['E_state_dict'])
G_global.load_state_dict(global_checkpoint['G_state_dict'], strict=False)

global_checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32)
D_global.load_state_dict(global_checkpoint['D_state_dict'])

global_lossesG = global_checkpoint['lossesG']
global_lossesD = global_checkpoint['lossesD']
global_optimizerG.load_state_dict(global_checkpoint['optimizerG'])
global_optimizerD.load_state_dict(global_checkpoint['optimizerD'])

# Set model to train mode
G_global.train()
E_global.train()
D_global.train()

G_server.train()
E_server.train()
D_server.train()

# Training
server_lossD = []
server_lossG = []
server_E_weights = [] 
server_D_weights = [] 
server_G_weights = []

def main_train():
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
            g_y = g_y.to(device) # 2, 3, 256, 256
            W_i = W_i.squeeze(-1).transpose(0,1).to(device).requires_grad_() # 2, 512, 1 --> 512, 2
            
            D_server.load_W_i(W_i)
	        
            with torch.autograd.enable_grad():
                #zero the parameter gradients
                server_optimizerG.zero_grad()
                server_optimizerD.zero_grad()

                #forward
                # Calculate average encoding vector for video
                f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxK,2,3,224,224

                e_vectors = E_server(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxK,512,1
                e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,K,512,1
                e_hat = e_vectors.mean(dim=1) # 2,512,1

                #train G and D
                x_hat = G_server(g_y, e_hat) # 2, 3, 256, 256

                r_hat, D_hat_res_list = D_server(x_hat, g_y, i) # 2x1x1 , 7

                with torch.no_grad():
                    r, D_res_list = D_server(x, g_y, i)

                lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list, e_vectors, D_server.W_i, i)

                lossG.backward(retain_graph=False)
                server_optimizerG.step()
        
            with torch.autograd.enable_grad():
                server_optimizerG.zero_grad()
                server_optimizerD.zero_grad()
                x_hat.detach_().requires_grad_()
                r_hat, D_hat_res_list = D_server(x_hat, g_y, i)
                lossDfake = criterionDfake(r_hat)

                r, D_res_list = D_server(x, g_y, i)
                lossDreal = criterionDreal(r)

                lossD = lossDfake + lossDreal
                lossD.backward(retain_graph=False)
                server_optimizerD.step()

                server_optimizerD.zero_grad()
                r_hat, D_hat_res_list = D_server(x_hat, g_y, i)
                lossDfake = criterionDfake(r_hat)

                r, D_res_list = D_server(x, g_y, i)
                lossDreal = criterionDreal(r)

                lossD = lossDfake + lossDreal
                lossD.backward(retain_graph=False)
                server_optimizerD.step()

            # Save discriminator weights
            for enum, idx in enumerate(i):
                torch.save({'W_i': D_server.W_i[:,enum].unsqueeze(-1)}, server_path_to_Wi+'/W_'+str(idx.item())+'/W_'+str(idx.item())+'.tar')

            # Save loss from batches each epoch
            batch_loss_D.append(lossD.item())
            batch_loss_G.append(lossG.item())
            batch_server_loss_G.append(lossG.item())
            batch_server_loss_D.append(lossD.item())

        # Save server losses and weights
        server_lossD.append(sum(batch_loss_D)/len(batch_loss_D))
        server_lossG.append(sum(batch_loss_G)/len(batch_loss_G))
        server_E_weights.append(copy.deepcopy(E_server.state_dict()))
        server_G_weights.append(copy.deepcopy(G_server.state_dict()))
        server_D_weights.append(copy.deepcopy(D_server.state_dict()))

        # Output training stats      
        print('Saving latest server model...')
        torch.save({
            'epoch': epoch+1,
            'lossesG': batch_server_loss_G,
            'lossesD': batch_server_loss_D,
            'E_state_dict': E_server.state_dict(),
            'G_state_dict': G_server.state_dict(),
            'D_state_dict': D_server.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizerG': server_optimizerG.state_dict(),
            'optimizerD': server_optimizerD.state_dict()
        }, server_path_to_chkpt)
        out = (x_hat[0]*255).transpose(0,2)
        out = out.type(torch.uint8).to("cpu").numpy()
        plt.imsave(server_training_image_path+str(epoch)+".png", out)
        print('...Done saving latest server model')

        if epoch%5 == 0:
            print('Saving latest server model...')
            torch.save({
                'epoch': epoch+1,
                'lossesG': batch_server_loss_G,
                'lossesD': batch_server_loss_D,
                'E_state_dict': E_server.state_dict(),
                'G_state_dict': G_server.state_dict(),
                'D_state_dict': D_server.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': i_batch,
                'optimizerG': server_optimizerG.state_dict(),
                'optimizerD': server_optimizerD.state_dict()
            }, server_path_to_backup+"_"+str(epoch)+".tar")
            print('...Done saving latest server model')

        # Start thread to run function which asks for local model when 5 epochs are completed
        if epoch%1 == 0:
            print("Check other model: ")
            thread_average = threading.Thread(target=average_models, args=())
            thread_average.daemon = True
            thread_average.start()

# Start thread to run main training 
thread = threading.Thread(target=main_train, args=())
thread.daemon = True # it doesn't care if program shuts down
thread.start()

@app.route('/average_models/', methods=['POST'])
def add_local_data():
    print("hola\n")
    r = request
    print(r)
    file = request.files['file']
    print(file)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))

    checkpoint_GATV = torch.load(os.path.join(UPLOAD_FOLDER, filename))
    GATV_E_weights.append(checkpoint_GATV['E_state_dict'])
    GATV_G_weights.append(checkpoint_GATV['G_state_dict'])
    GATV_D_weights.append(checkpoint_GATV['D_state_dict']) 
    GATV_lossG.append(checkpoint_GATV["lossesG"]) 
    GATV_lossD.append(checkpoint_GATV["lossesD"])

    response = {'message': 'OK!'}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':

	app.run(host="192.168.0.47", port=5000)