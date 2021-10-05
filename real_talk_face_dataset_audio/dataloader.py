#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:56 2020

@author: pati
"""

# Needed libraries and packages
import os
import torch

import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from testing_utils import generate_cropped_landmarks, select_finetuning_frames, get_testing_spectrogram, select_images_frames
from utils import select_frames, get_landmarks, get_spectrogram



class CelebDataset(Dataset):
    def __init__(self, path_list, K, device, path_to_Wi_images, path_to_Wi_audio):

        self.K = K
        self.device = device
        self.path_list = path_list
        self.path_to_Wi_images = path_to_Wi_images
        self.path_to_Wi_audio = path_to_Wi_audio

        # List with video paths
        self.video_paths = []

        video_number = 0
        for item in path_list:
            path = Path(item)
            for celeb_folder in path.iterdir():
                for video_folder in celeb_folder.iterdir():
                    video_number += 1
                    for txt_folder in video_folder.iterdir():
                        self.video_paths.append(video_folder)

        # Dataset length --> number of videos
        self.len = video_number

    def __len__(self):
        return self.len
    
    # This function should return K random frames with features from a video sequence
    def __getitem__(self, idx):

        shots = self.K
        video_path = self.video_paths[idx]
        frames_paths = select_frames(self.K, video_path) # 8

        if len(frames_paths)<self.K:
            shots = len(frames_paths)

        #print("Video path:")
        #print(video_path)
        #print("Frames_paths: ")
        #print(frames_paths)

        # From each frame, take image features and return them
        cropped_frames_landmarks = get_landmarks(video_path, frames_paths)
        cropped_frames_landmarks = torch.from_numpy(np.array(cropped_frames_landmarks)).type(dtype = torch.float)
        cropped_frames_landmarks = cropped_frames_landmarks.transpose(2,4).to(self.device)/255 # K,2,3,256,256

        g_idx = torch.randint(low = 0, high = shots, size = (1,1))
        x = cropped_frames_landmarks[g_idx,0].squeeze() # squeeze --> removes dimensions of size 1 from the shape of a tensor # 3, 256, 256
        g_y = cropped_frames_landmarks[g_idx,1].squeeze() # 3, 256, 256

        # Get the spectrogram from g_idx
        g_spect = get_spectrogram(video_path, frames_paths[g_idx]) # (256, 256, 3)
        g_spect = torch.from_numpy(np.array(g_spect)).type(dtype = torch.float) # torch.Size([256, 256, 3])
        g_spect = g_spect.transpose(0,2).to(self.device)/255 # torch.Size([3, 256, 256])

        if self.path_to_Wi_images is not None and self.path_to_Wi_audio is not None:
            try:
                W_i_images = torch.load(self.path_to_Wi_images+'/W_'+str(idx)+'/W_'+str(idx)+'.tar', map_location='cpu')['W_i_images'].requires_grad_(False) # torch.Size([512, 1])
                W_i_audio = torch.load(self.path_to_Wi_audio+'/W_'+str(idx)+'/W_'+str(idx)+'.tar', map_location='cpu')['W_i_audio'].requires_grad_(False) # torch.Size([512, 1])
            except:
                print("[ERROR] Loading: ", self.path_to_Wi_images+'/W_'+str(idx)+'/W_'+str(idx)+'.tar')
                W_i_images = torch.rand((512,1))
                W_i_audio = torch.rand((512,1))
        else:
            W_i_images = None
            W_i_audio = None
        
        return cropped_frames_landmarks, x, g_y, idx, W_i_images, W_i_audio, g_spect
                

class AudioFineTuningVideoDataset(Dataset):
    def __init__(self, path_to_video, device, face_aligner, path_to_audio, padding):
        self.path_to_video = path_to_video
        self.device = device
        self.face_aligner = face_aligner
        self.path_to_audio = path_to_audio
        self.padding = padding
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        path = self.path_to_video
        frame_has_face = False
        while not frame_has_face:
            try:
                frame_mark, audio_list, fps = select_finetuning_frames(path, 1)
                frame_mark = generate_cropped_landmarks(frame_mark, self.face_aligner, pad=self.padding)
                g_spect = get_testing_spectrogram(self.path_to_audio, audio_list[0], fps) # (256, 256, 3)
                frame_has_face = True
            except:   
                print('No face or spectrogram detected, retrying')

        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #1,2,256,256,3
        frame_mark = frame_mark.transpose(2,4).to(self.device) #1,2,3,256,256
        
        x = frame_mark[0,0].squeeze()/255
        g_y = frame_mark[0,1].squeeze()/255

        g_spect = torch.from_numpy(np.array(g_spect)).type(dtype = torch.float) # torch.Size([256, 256, 3])
        g_spect = g_spect.transpose(0,2).to(self.device)/255 # torch.Size([3, 256, 256])

        return x, g_y, g_spect

        