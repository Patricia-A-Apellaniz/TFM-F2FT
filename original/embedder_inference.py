"""Main"""
import torch

from dataset.video_extraction_conversion import select_frames, select_images_frames, generate_cropped_landmarks, generate_landmarks
from network.blocks import *
from network.model import Embedder

import numpy as np


"""Hyperparameters and config"""
device = torch.device("cuda:0")
cpu = torch.device("cpu")
path_to_e_hat_video = 'e_hat_video.tar'
path_to_e_hat_images = 'e_hat_images.tar'
path_to_chkpt = 'model_weights.tar'
path_to_video = 'examples/fine_tuning/pedro.mp4'
path_to_images = 'examples/fine_tuning/test_images'
T = 32



"""Loading Embedder input"""
print("Selecting video frames...")
frame_mark_video = select_frames(path_to_video , T)
print("Generating cropped landmarks...")
frame_mark_video = generate_cropped_landmarks(frame_mark_video, pad=50)
#print("Generating landmarks...")
#frame_mark_video = generate_landmarks(frame_mark_video)
print("Transforming...")
frame_mark_video = torch.from_numpy(np.array(frame_mark_video)).type(dtype = torch.float) #T,2,256,256,3
#print("frame_mark_video: ", np.shape(frame_mark_video))
frame_mark_video = frame_mark_video.transpose(2,4).to(device) #T,2,3,256,256
#print("frame_mark_video: ", np.shape(frame_mark_video))
f_lm_video = frame_mark_video.unsqueeze(0) #1,T,2,3,256,256
#print("f_lm_video: ", np.shape(f_lm_video))

print("Selecting image frames...")
frame_mark_images = select_images_frames(path_to_images)
print("Generating cropped landmarks...")
frame_mark_images = generate_cropped_landmarks(frame_mark_images, pad=50)
#print("Generating landmarks...")
#frame_mark_images = generate_cropped_landmarks(frame_mark_images)
print("Transforming...")
frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(dtype = torch.float) #T,2,256,256,3
#print("frame_mark_images: ", np.shape(frame_mark_images))
frame_mark_images = frame_mark_images.transpose(2,4).to(device) #T,2,3,256,256
#print("frame_mark_images: ", np.shape(frame_mark_images))
f_lm_images = frame_mark_images.unsqueeze(0) #1,T,2,3,256,256
#print("f_lm_images: ", np.shape(f_lm_images))


print("Initializating embedder...")
E = Embedder(256).to(device)
E.eval()


"""Loading from past checkpoint"""
print("Loading checkpoint...")
checkpoint = torch.load(path_to_chkpt, map_location=cpu)
E.load_state_dict(checkpoint['E_state_dict'])


"""Inference"""
with torch.no_grad():
    #forward
    # Calculate average encoding vector for video
    print("Calculating embedding vector for video...")
    f_lm = f_lm_video
    f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxT,2,3,224,224
    #print("f_lm_compact: ", np.shape(f_lm_compact))
    e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxT,512,1
    #print("e_vectors: ", np.shape(e_vectors))
    e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,T,512,1
    #print("e_vectors: ", e_vectors)
    #print("e_vectors: ", np.shape(e_vectors))
    e_hat_video = e_vectors.mean(dim=1)
    #print("e_hat_video: ", np.shape(e_hat_video))
    
    print("Calculating embedding vector for images...")
    f_lm = f_lm_images
    f_lm_compact = f_lm.view(-1, f_lm.shape[-4], f_lm.shape[-3], f_lm.shape[-2], f_lm.shape[-1]) #BxT,2,3,224,224
    #print("f_lm_compact: ", np.shape(f_lm_compact))
    e_vectors = E(f_lm_compact[:,0,:,:,:], f_lm_compact[:,1,:,:,:]) #BxT,512,1
    #print("e_vectors: ", np.shape(e_vectors))
    e_vectors = e_vectors.view(-1, f_lm.shape[1], 512, 1) #B,T,512,1
    #print("e_vectors: ", np.shape(e_vectors))
    e_hat_images = e_vectors.mean(dim=1)
    #print("e_hat_images: ", np.shape(e_hat_images))


print('Saving e_hat...')
torch.save({
        'e_hat': e_hat_video
        }, path_to_e_hat_video)
torch.save({
        'e_hat': e_hat_images
        }, path_to_e_hat_images)
print('...Done saving')