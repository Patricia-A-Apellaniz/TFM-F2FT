#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 17:11 2020

@author: pati
"""

# Needed libraries and packages
import os
import ast
import cv2
import torch
import argparse
import subprocess
import face_alignment

import numpy as np
import torch.optim as optim

from datetime import datetime
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from params import *
from network.model import *
from network.blocks import *
from loss.loss_generator import *
from loss.loss_discriminator import *
from dataloader import AudioFineTuningVideoDataset
from testing_utils import select_frames, generate_cropped_landmarks, generate_landmarks_and_spectrogram


# Construct and Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video","--videoPath", type = str, 
    help = "path to video")
ap.add_argument("-videoEmbedding","--videoPathEmbedding", default = "embedding_examples/Pedro.mp4", type = str, 
    help = "path to embedding video")
ap.add_argument("-K","--numberOfShots", default = 32, type = int, 
    help = "number of shots for image embedding vector")
ap.add_argument("-output", "--outputPath", default = "testing/", type = str, 
    help = 'path to the output directory, where output files are stored.')
ap.add_argument("-finetuning","--finetuningEpochs", default = 200, type = int, 
    help = "number of epochs for finetuning training")
ap.add_argument("-pad","--paddingImages", default = 50, type = int, 
    help = "padding incropped images")


args = vars(ap.parse_args())


# Check if gpu available
gpu = torch.cuda.is_available()
if gpu:
    torch.cuda.set_device("cuda:0") 
device = 'cuda:0' if gpu else 'cpu'
cpu = torch.device("cpu")

# Global configuration
output_path = args["outputPath"] + "/"
path_to_output_video = args["videoPath"]
path_to_e_hat_embedding = output_path + "e_hat_video.tar"
path_to_chkpt = server_path_to_chkpt
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
path_to_embedding_video = args["videoPathEmbedding"]
finetuning = args["finetuningEpochs"]
K = args["numberOfShots"]
padding = args["paddingImages"]
face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

if not os.path.exists(output_path):
	os.mkdir(output_path)


## IMAGE EMBEDDING VECTOR ##
def embedder_inference():
	# Preparing landmarks and frames
	print("Selecting video frames...")
	frame_mark_video = select_frames(path_to_embedding_video, K)
	print("Generating cropped landmarks...")
	frame_mark_video = generate_cropped_landmarks(frame_mark_video, pad=padding, face_aligner=face_aligner)
	print("Transforming...")
	frame_mark_video = torch.from_numpy(np.array(frame_mark_video)).type(dtype = torch.float) #T,2,256,256,3
	frame_mark_video = frame_mark_video.transpose(2,4).to(device)/255 #torch.Size([32, 2, 3, 256, 256])
	f_lm_video = frame_mark_video.unsqueeze(0) #torch.Size([1, 32, 2, 3, 256, 256])

	print("Initializating image embedder...")
	E_image = ImageEmbedder(256).to(device)
	E_image.eval()
	E_image.load_state_dict(checkpoint['image_E_state_dict'])

	# Inference
	with torch.no_grad():
	    # Calculate average encoding vector for video
	    print("Calculating embedding vector for video...")
	    f_lm = f_lm_video
	    f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxT,2,3,224,224
	    e_vectors = E_image(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxT,512,1
	    e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,T,512,1
	    e_hat_video = e_vectors.mean(dim=1)

	# Saving output
	print('Saving e_hat...')
	torch.save({
	        'e_hat': e_hat_video
	        }, path_to_e_hat_embedding)
	print('...Done saving')


print("Starting image embedding process...")
embedder_inference()


# Extracting audio from video
path_to_embedding_audio_test = str(path_to_output_video).split(".")[-2] + ".aac"
if not os.path.exists(path_to_embedding_audio_test):
    print("Extracting audio from video...")
    command_audio = ["ffmpeg",
    "-i",
    path_to_output_video,                   
    "-vn",
    path_to_embedding_audio_test]
    run = subprocess.run(command_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

path_to_save = output_path+'finetuned_model.tar'
## FINETUNNING ## 
def finetuning_training():

	print("Extracting audio from finetunning audio...")
	# Extracting audio from video
	path_to_embedding_audio_finetuning = str(path_to_embedding_video).split(".")[-2] + ".aac"
	if not os.path.exists(path_to_embedding_audio_finetuning):
		print("Extracting audio from video...")
		command_audio = ["ffmpeg",
		"-i",
		path_to_embedding_video,                   
		"-vn",
		path_to_embedding_audio_finetuning]
		run = subprocess.run(command_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	# Configuration
	display_training = True
	finetuning_epochs = args["finetuningEpochs"]
	if not display_training:
	    matplotlib.use('agg')
	dataset = AudioFineTuningVideoDataset(path_to_embedding_video, device, face_aligner, path_to_embedding_audio_finetuning, padding)
	dataLoader = DataLoader(dataset, batch_size=2, shuffle=False)

	# Initializing embedding vector
	print("Initializating image embedder...")
	e_hat_video = torch.load(path_to_e_hat_embedding, map_location=cpu)
	e_hat_video = e_hat_video['e_hat']

	# Initializing models
	print("Initializating generator and discriminator...")
	G = Generator(256)
	D = Discriminator(dataset.__len__(), server_path_to_Wi)
	G.train()
	D.train()

	optimizerG = optim.Adam(params = G.parameters(), lr=5e-5)
	optimizerD = optim.Adam(params = D.parameters(), lr=2e-4)

	print("Initializating audio embedder...")
	E_audio = AudioEmbedder(256).to(device)
	E_audio.train()

	# Criterion
	criterionG = LossGF(VGGFace_body_path='Pytorch_VGGFACE_IR.py',
	                   VGGFace_weight_path='Pytorch_VGGFACE.pth', device=device)
	criterionDreal = LossDSCreal()
	criterionDfake = LossDSCfake()

	# Training init
	epochCurrent = epoch = i_batch = 0
	lossesG = []
	lossesD = []
	i_batch_current = 0

	# Loading from past checkpoint
	checkpoint['D_state_dict']['W_i'] = torch.rand(512, 32) # change W_i for finetuning
	G.load_state_dict(checkpoint['G_state_dict'])
	D.load_state_dict(checkpoint['D_state_dict'], strict = False)
	D.finetuning_init(finetuning=True, e_finetuning=e_hat_video)
	E_audio.load_state_dict(checkpoint['audio_E_state_dict'])
	G.to(device)
	D.to(device)
	E_audio.to(device)
	e_hat_video = e_hat_video.to(device)

	# Training
	print("Finetuning training starting...")
	batch_start = datetime.now()
	for epoch in range(finetuning):
		for i_batch, (x, g_y, g_spect) in enumerate(dataLoader):
			with torch.autograd.enable_grad():

				print("Calculating embedding vector for audio...")
				audio_vector = E_audio(g_spect)
				e_hat = torch.cat((e_hat_video, audio_vector), 1)

				# Update embedding vector with new audio
				G.finetuning_init(finetuning=True, e_finetuning=e_hat)

				# Zero the parameter gradients
				optimizerG.zero_grad()
				optimizerD.zero_grad()

				# Forward
				# Train G and D
				x_hat = G(g_y, e_hat)
				r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
				with torch.no_grad():
					r, D_res_list = D(x, g_y, i=0)

				lossG = criterionG(x, x_hat, r_hat, D_res_list, D_hat_res_list)
				lossG.backward(retain_graph=False)
				optimizerG.step()

				# Train D
				optimizerD.zero_grad()
				x_hat.detach_().requires_grad_()
				r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
				r, D_res_list = D(x, g_y, i=0)

				lossDfake = criterionDfake(r_hat)
				lossDreal = criterionDreal(r)

				lossD = lossDreal + lossDfake
				lossD.backward(retain_graph=False)
				optimizerD.step() 

				# Train D again
				optimizerG.zero_grad()
				optimizerD.zero_grad()
				r_hat, D_hat_res_list = D(x_hat, g_y, i=0)
				r, D_res_list = D(x, g_y, i=0)

				lossDfake = criterionDfake(r_hat)
				lossDreal = criterionDreal(r)

				lossD = lossDreal + lossDfake
				lossD.backward(retain_graph=False)
				optimizerD.step()

				lossesD.append(lossD.item())
				lossesG.append(lossG.item())
    	    
			# Output training stats
			if epoch % 10 == 0:
				batch_end = datetime.now()
				avg_time = (batch_end - batch_start) / 10
				print('\n\n[INFO] avg batch time for batch size of', x.shape[0],':',avg_time)
				batch_start = datetime.now()
				print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(y)): %.4f'
					% (epoch, finetuning, i_batch, len(dataLoader),
						lossD.item(), lossG.item(), r.mean(), r_hat.mean()))

	plt.clf()
	plt.plot(lossesG, label="Losses Generator") #blue
	plt.plot(lossesD, label="Losses Discriminator") #orange
	plt.title('Fine-tunning stage Losses values')
	plt.xlabel('Iterations')
	plt.ylabel('Values')
	plt.legend(loc='upper right', frameon=True)
	plt.show()       
	        

	print('Saving finetuned model...')
	torch.save({
	        'epoch': epoch,
	        'lossesG': lossesG,
	        'lossesD': lossesD,
	        'G_state_dict': G.state_dict(),
	        'D_state_dict': D.state_dict(),
	        'audio_E_state_dict': E_audio.state_dict(),
	        'optimizerG_state_dict': optimizerG.state_dict(),
	        'optimizerD_state_dict': optimizerD.state_dict(),
	        }, path_to_save)
	print('...Done saving latest')

if finetuning>0:
	print("Starting finetuning training...")
	finetuning_training()



## VIDEO INFERENCE ##
def video_inference():
	if finetuning>0:
		print("Loading finetuned model...")
		path_to_chkpt = path_to_save
		checkpoint = torch.load(path_to_chkpt, map_location=cpu) 
	else:
		path_to_chkpt = server_path_to_chkpt
		checkpoint = torch.load(path_to_chkpt, map_location=cpu)
	
	# Loading from checkpoint
	e_hat_video = torch.load(path_to_e_hat_embedding, map_location=cpu)
	e_hat_video = e_hat_video['e_hat'].to(device) # torch.Size([1, 512, 1])

	print("Initializing generator...")
	G = Generator(256)
	G.to(device)
	G.eval()
	G.load_state_dict(checkpoint['G_state_dict'])

	print("Initializating audio embedder...")
	E_audio = AudioEmbedder(256).to(device)
	E_audio.eval()
	E_audio.load_state_dict(checkpoint['audio_E_state_dict'])

	print("Loading video...")
	cap = cv2.VideoCapture(path_to_output_video)
	n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))
	size = (256*3,256)
	video = cv2.VideoWriter(output_path+"video_generated.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, size)

	ret = True
	with torch.no_grad():
		frame_count = 0
		while ret:
			x, g_y, ret, g_spect, frame_count = generate_landmarks_and_spectrogram(cap=cap, fps=fps, device=device, pad=padding, audio_path=path_to_embedding_audio_test, fa=face_aligner, frame_count=frame_count)
			if ret:
				x = x.unsqueeze(0)/255
				g_y = g_y.unsqueeze(0)/255
				g_spect = g_spect.unsqueeze(0)/255

				print("Calculating embedding vector for audio...")
				audio_vector = E_audio(g_spect)
				e_hat = torch.cat((e_hat_video, audio_vector), 1)

				if finetuning:
					G.finetuning_init(finetuning=True, e_finetuning=e_hat)

				x_hat = G(g_y, e_hat)

				out1 = x_hat.transpose(1,3)[0]
				out1 = out1.to(cpu).numpy()

				out2 = x.transpose(1,3)[0]
				out2 = out2.to(cpu).numpy()

				out3 = g_y.transpose(1,3)[0]
				out3 = out3.to(cpu).numpy()

				me = cv2.cvtColor(out2*255, cv2.COLOR_BGR2RGB)
				landmark = cv2.cvtColor(out3*255, cv2.COLOR_BGR2RGB)
				fake = cv2.cvtColor(out1*255, cv2.COLOR_BGR2RGB)

				img = np.concatenate((me, landmark, fake), axis=1)
				img = img.astype("uint8")
				if frame_count == 330:
					img_saved = img
				video.write(img)

				frame_count+=1
				print("Frame count: ", frame_count, "/", n_frames)

	cap.release()
	video.release()
	plt.imshow(cv2.cvtColor(img_saved,cv2.COLOR_BGR2RGB))
	plt.show()


print("Starting video inference process...")
video_inference()










