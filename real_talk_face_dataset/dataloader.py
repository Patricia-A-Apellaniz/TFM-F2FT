import torch
import os
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from utils import select_training_frames, get_landmarks
from testing_utils import *

class VidDataSet(Dataset):
    def __init__(self, K, path_list, device, path_to_Wi):
        self.K = K
        self.device = device
        self.path_list = path_list
        self.path_to_Wi = path_to_Wi

        # Lists with celeb id paths, video paths, txt paths
        self.celeb_paths = []
        self.video_paths = []
        self.txt_paths = []

        video_number = 0
        for item in path_list:
            path = Path(item)
            for celeb_folder in path.iterdir():
                for video_folder in celeb_folder.iterdir():
                    video_number += 1
                    for txt_folder in video_folder.iterdir():
                        self.celeb_paths.append(celeb_folder)
                        self.video_paths.append(video_folder)
                        self.txt_paths.append(txt_folder)

        # Dataset length --> number of videos
        self.len = video_number

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        shots = self.K
        video_path = self.video_paths[idx]
        frames_paths = select_training_frames(self.K, video_path) # 8

        if len(frames_paths)<self.K:
            shots = len(frames_paths)

        # From each frame, take image features and return them
        cropped_frames_landmarks = get_landmarks(video_path, frames_paths)
        cropped_frames_landmarks = torch.from_numpy(np.array(cropped_frames_landmarks)).type(dtype = torch.float)
        cropped_frames_landmarks = cropped_frames_landmarks.transpose(2,4).to(self.device)/255 # K,2,3,256,256

        g_idx = torch.randint(low = 0, high = shots, size = (1,1))
        x = cropped_frames_landmarks[g_idx,0].squeeze() # squeeze --> removes dimensions of size 1 from the shape of a tensor # 3, 256, 256
        g_y = cropped_frames_landmarks[g_idx,1].squeeze() # 3, 256, 256

        if self.path_to_Wi is not None:
            try:
                W_i = torch.load(self.path_to_Wi+'/W_'+str(idx)+'/W_'+str(idx)+'.tar', map_location='cpu')['W_i'].requires_grad_(False) # torch.Size([512, 1])
            except:
                print("[ERROR] Loading: ", self.path_to_Wi+'/W_'+str(idx)+'/W_'+str(idx)+'.tar')
                W_i = torch.rand((512,1))
        else:
            W_i = None
        
        return cropped_frames_landmarks, x, g_y, idx, W_i


class FineTuningImagesDataset(Dataset):
    def __init__(self, path_to_images, device, face_aligner, padding):
        self.path_to_images = path_to_images
        self.device = device
        self.face_aligner = face_aligner
        self.padding = padding
    
    def __len__(self):
        return len(os.listdir(self.path_to_images))
    
    def __getitem__(self, idx):
        frame_mark_images = select_images_frames(self.path_to_images)
        random_idx = torch.randint(low = 0, high = len(frame_mark_images), size = (1,1))
        frame_mark_images = [frame_mark_images[random_idx]]
        frame_mark_images = generate_cropped_landmarks(frame_mark_images, self.face_aligner, pad=self.padding)
        frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(dtype = torch.float) #1,2,256,256,3
        frame_mark_images = frame_mark_images.transpose(2,4).to(self.device) #1,2,3,256,256
        
        x = frame_mark_images[0,0].squeeze()/255
        g_y = frame_mark_images[0,1].squeeze()/255
        
        return x, g_y
        

class FineTuningVideoDataset(Dataset):
    def __init__(self, path_to_video, device, face_aligner, padding):
        self.path_to_video = path_to_video
        self.device = device
        self.face_aligner = face_aligner
        self.padding = padding
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        path = self.path_to_video
        frame_has_face = False
        while not frame_has_face:
            try:
                frame_mark = select_frames(path , 1)
                frame_mark = generate_cropped_landmarks(frame_mark, self.face_aligner, pad=self.padding)
                frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #1,2,256,256,3
                frame_mark = frame_mark.transpose(2,4).to(self.device) #1,2,3,256,256
                frame_has_face = True
            except:
                print('No face detected, retrying')
        
        
        x = frame_mark[0,0].squeeze()/255
        g_y = frame_mark[0,1].squeeze()/255
        return x, g_y
