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
from testing_utils import select_images_frames, generate_cropped_landmarks, generate_landmarks_and_spectrogram


# Construct and Parse input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-video","--videoPath", type = str, 
    help = "path to video")
ap.add_argument("-imagesEmbedding","--imagesPathEmbedding", default = "embedding_examples/test_images/", type = str, 
    help = "path to embedding video")
ap.add_argument("-output", "--outputPath", type = str, 
    help = 'path to the output directory, where output files are stored.')
ap.add_argument('-device', '--device', default = 'cuda:0', type = str,
	help = 'where to run program')
ap.add_argument("-pad","--paddingImages", default = 200, type = int, 
    help = "padding incropped images")

args = vars(ap.parse_args())

# Global configuration
device = torch.device(args["device"])
cpu = torch.device("cpu")
output_path = args["outputPath"] + "/"
path_to_video = args["videoPath"]
path_to_e_hat_embedding = output_path + "e_hat_images.tar"
path_to_chkpt = server_path_to_chkpt
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
path_to_embedding_images = args["imagesPathEmbedding"]
padding = args["paddingImages"]
face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=args["device"])

if not os.path.exists(output_path):
	os.mkdir(output_path)

## IMAGE EMBEDDING VECTOR ##
def embedder_inference():
	# Preparing landmarks and frames
	print("Selecting video frames...")
	frame_mark_video = select_images_frames(path_to_embedding_images)
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
path_to_embedding_audio_test = str(path_to_video).split(".")[-2] + ".aac"
if not os.path.exists(path_to_embedding_audio_test):
    print("Extracting audio from video...")
    command_audio = ["ffmpeg",
    "-i",
    path_to_video,                   
    "-vn",
    path_to_embedding_audio_test]
    run = subprocess.run(command_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


## VIDEO INFERENCE ##
def video_inference():
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
	cap = cv2.VideoCapture(path_to_video)
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










