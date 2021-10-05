#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 10:09 2020

@author: pati
"""

# Needed libraries and packages
import os
import cv2
import math
import shutil
import random
import librosa
import subprocess
import librosa.display

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from scipy.io import loadmat
from pydub import AudioSegment
from tensorboardX import SummaryWriter
from librosa.feature import melspectrogram


# This function mounts a virtual link to a directory in other device with data
def mount_server_disks(mount_dir_visiona1, mount_dir_visiona2):
	ip_visiona1 = "192.168.0.13"
	user_pc_visiona1 = "visiona"

	ip_visiona2 = "192.168.0.15"
	user_pc_visiona2 = "visiona"

	dataset_dir_visiona1 = "/home/visiona/Desktop/pati/preprocessed"
	dataset_dir_visiona2 = "/home/visiona/Escritorio/pati/preprocessed"

	try:
	    os.system("sudo umount "+mount_dir_visiona1)
	    os.system("sudo umount "+mount_dir_visiona2)
	except:
	    pass

	if not os.path.exists(mount_dir_visiona1) or os.path.exists(mount_dir_visiona2):
	    os.system("sudo mkdir "+mount_dir_visiona1)
	    os.system("sudo mkdir "+mount_dir_visiona2)

	os.system("sudo sshfs -o allow_other "+user_pc_visiona1+"@"+ip_visiona1+":"+dataset_dir_visiona1+" "+mount_dir_visiona1)
	os.system("sudo sshfs -o allow_other "+user_pc_visiona2+"@"+ip_visiona2+":"+dataset_dir_visiona2+" "+mount_dir_visiona2)


# This function mounts a virtual link to a directory in other device with data
def mount_gatv_disks(mount_dir_famous, mount_dir_tfg_tfm):
    ip_tfg_tfm = "192.168.0.62"
    user_pc_tfg_tfm = "phoenix"

    ip_famoso = "192.168.0.63"
    user_pc_famoso = "gatv-reco"

    dataset_dir_tfg_tfm = "/home/phoenix/Desktop/pati/preprocessed"
    dataset_dir_famoso = "/home/gatv-reco/Desktop/pati/preprocessed"

    try:
        os.system("sudo umount "+mount_dir_famous)
        os.system("sudo umount "+mount_dir_tfg_tfm)
    except:
        pass

    if not os.path.exists(mount_dir_famous) or os.path.exists(mount_dir_tfg_tfm):
        os.system("sudo mkdir "+mount_dir_famous)
        os.system("sudo mkdir "+mount_dir_tfg_tfm)

    os.system("sudo sshfs -o allow_other "+user_pc_famoso+"@"+ip_famoso+":"+dataset_dir_famoso+" "+mount_dir_famous)
    os.system("sudo sshfs -o allow_other "+user_pc_tfg_tfm+"@"+ip_tfg_tfm+":"+dataset_dir_tfg_tfm+" "+mount_dir_tfg_tfm)


# This function creates necessary folders
def define_directories(reset_model_info, root):
    root_path = root
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    elif reset_model_info:
        shutil.rmtree(root_path, ignore_errors=True)
        os.mkdir(root_path)

    logs_path = root_path+"/logs"
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    elif reset_model_info:
        shutil.rmtree(logs_path, ignore_errors=True)
        os.mkdir(logs_path)

    logger = SummaryWriter(logs_path)

    wi_path = root_path+"/wi_weights"
    if not os.path.exists(wi_path):
        os.mkdir(wi_path)
    elif reset_model_info:
        shutil.rmtree(wi_path, ignore_errors=True)
        os.mkdir(wi_path)

    back_up_path = root_path+"/weights"
    if not os.path.exists(back_up_path):
        os.mkdir(back_up_path)
    elif reset_model_info:
        shutil.rmtree(back_up_path, ignore_errors=True)
        os.mkdir(back_up_path)

    images_folder = root_path+"/training_images"
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)
    elif reset_model_info:
        shutil.rmtree(images_folder, ignore_errors=True)
        os.mkdir(images_folder)

    return logger


# This function creates necessary folders
def define_global_directories():
    global_folder = "global"
    if not os.path.exists(global_folder):
        os.mkdir(global_folder)

    global_back_up_path = "global/weights"
    if not os.path.exists(global_back_up_path):
        os.mkdir(global_back_up_path)

    images_folder = "global/training_images"
    if not os.path.exists(images_folder):
        os.mkdir(images_folder)



def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring (fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h,4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf, 3, axis = 2)
    return buf

def fig2img (fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


# This function groups all video frames separated by txt in one list and selects K random ones
def select_frames(K, video_path):
    frames_paths = []

    for txt_folder in video_path.iterdir():
        path = Path(os.path.join(txt_folder, "frames"))
        for frame in path.iterdir():
            frames_paths.append(frame)

    # If there are not enough frames in the video
    if K>len(frames_paths):
        print("[Warning] K > number frames: ")
        print(video_path)
        K = len(frames_paths)

    random_frames_paths = random.choices(frames_paths, k=K)

    return random_frames_paths


# This function loads cropped frames and landmarks given path
def get_landmarks(video_path, frames_paths):
    frames_landmarks = []
    cropped_frames_landmarks = []
    for item in frames_paths:

        frame = str(item).split("/")[-1]
        filename, file_extension = os.path.splitext(frame)
        txt_folder = str(item).split("/")[-3]

        # Cropped Frames and Cropped Landmarks
        cropped_frame_path = os.path.join(video_path, txt_folder+"/crops/"+frame)
        try:
            cropped_image = cv2.cvtColor(np.array(cv2.imread(cropped_frame_path))[:,:,:3], cv2.COLOR_RGB2BGR)
        except:
            print("[ERROR] Crop could not be read:")
            print(cropped_frame_path)
            continue

        landmarks_cropped_path = os.path.join(video_path, txt_folder+"/landmarks_crops/"+frame)
        cropped_landmarks = cv2.cvtColor(np.array(cv2.imread(landmarks_cropped_path))[:,:,:3], cv2.COLOR_RGB2BGR)

        # Cropped frames have different sizes --> reshape to 256x256
        cropped_image = cv2.resize(cropped_image, (256, 256))
        cropped_landmarks = cv2.resize(cropped_landmarks, (256, 256))
        cropped_frames_landmarks.append((cropped_image, cropped_landmarks))

    return cropped_frames_landmarks


# This function loads spectrograms
def get_spectrogram(video_path, frame_path):

	## En el embedder de audio no deberían entrar K shots de spectrograma
	# Deben entrar 1 a 1 (En el autoencoder entra 1 imagen, pues 1 spectrograma relativo a una imagen)
	# Combinar ambas salidas de los embedders --> mirar adaIN
	# Mejor que cambiar red, modificar vectores salida embeders?

    frame = str(frame_path).split("/")[-1]
    filename, file_extension = os.path.splitext(frame)
    txt_folder = str(frame_path).split("/")[-3]

    spectrogram_path = os.path.join(video_path, txt_folder+"/spectrogram/"+filename+".mat")
    frame_spectrogram = loadmat(spectrogram_path)
    fig_spect = plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(frame_spectrogram["spect"], ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=44100, fmax=20000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    im_spect = fig2img(fig_spect)
    plt.close(fig_spect)  
    plt.close('all')

    np_im_spect = cv2.cvtColor(np.array(im_spect)[:,:,:3], cv2.COLOR_RGB2BGR)
    frame_spect_resized = cv2.resize(np_im_spect, (256, 256))

    return frame_spect_resized
    