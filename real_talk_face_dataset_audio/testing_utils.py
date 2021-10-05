#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 19:07 2020

@author: pati
"""

# Needed libraries and packages
import os
import cv2
import math
import torch
import random
import librosa
import face_alignment
import librosa.display

import numpy as np
import soundfile as s

from utils import fig2img
from pydub import AudioSegment
from matplotlib import pyplot as plt
from librosa.feature import melspectrogram


def get_borders(preds):
    minX = maxX = preds[0,0]
    minY = maxY = preds[0,1]
    
    for i in range(1, len(preds)):
        x = preds[i,0]
        if x < minX:
            minX = x
        elif x > maxX:
            maxX = x
        
        y = preds[i,1]
        if y < minY:
            minY = y
        elif y > maxY:
            maxY = y
    
    return minX, maxX, minY, maxY


def crop_and_reshape_preds(preds, pad, out_shape=256):
    minX, maxX, minY, maxY = get_borders(preds)
    
    delta = max(maxX - minX, maxY - minY)
    deltaX = (delta - (maxX - minX))/2
    deltaY = (delta - (maxY - minY))/2
    
    deltaX = int(deltaX)
    deltaY = int(deltaY)
    
    # Crop
    for i in range(len(preds)):
        preds[i][0] = max(0, preds[i][0] - minX + deltaX + pad)
        preds[i][1] = max(0, preds[i][1] - minY + deltaY + pad)
    
    # Find reshape factor
    r = out_shape/(delta + 2*pad)
        
    for i in range(len(preds)):
        preds[i,0] = int(r*preds[i,0])
        preds[i,1] = int(r*preds[i,1])
    return preds


def crop_and_reshape_img(img, preds, pad, out_shape=256):
    minX, maxX, minY, maxY = get_borders(preds)
    
    # Find reshape factor
    delta = max(maxX - minX, maxY - minY)
    deltaX = (delta - (maxX - minX))/2
    deltaY = (delta - (maxY - minY))/2
    
    minX = int(minX)
    maxX = int(maxX)
    minY = int(minY)
    maxY = int(maxY)
    deltaX = int(deltaX)
    deltaY = int(deltaY)
    
    lowY = max(0,minY-deltaY-pad)
    lowX = max(0, minX-deltaX-pad)
    img = img[lowY:maxY+deltaY+pad, lowX:maxX+deltaX+pad, :]
    img = cv2.resize(img, (out_shape,out_shape))
    
    return img


def select_frames(video_path, K):
    cap = cv2.VideoCapture(video_path)
    
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Embedding video path: ", video_path)
    print("Number of total frames: ", n_frames)
    
    # There are not enough frames in the video
    if n_frames <= K: 
        rand_frames_idx = [1]*n_frames
    else:
        rand_frames_idx = [0]*n_frames
        i = 0
        while(i < K):
            idx = random.randint(0, n_frames-1)
            if rand_frames_idx[idx] == 0:
                rand_frames_idx[idx] = 1
                i += 1
    
    frames_list = []
    
    # Read until video is completed or no frames needed
    ret = True
    frame_idx = 0
    while(ret and frame_idx < n_frames):
        ret, frame = cap.read()  
        if ret and rand_frames_idx[frame_idx] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame)
            
        frame_idx += 1

    cap.release()

    return frames_list


def generate_cropped_landmarks(frames_list, face_aligner, pad=100):
    frame_landmark_list = []
    fa = face_aligner
    
    for i in range(len(frames_list)):
        try:
            input = frames_list[i]
            preds = fa.get_landmarks(input)[0]
            
            input = crop_and_reshape_img(input, preds, pad=pad)
            preds = crop_and_reshape_preds(preds, pad=pad)

            dpi = 100
            fig = plt.figure(figsize=(input.shape[1]/dpi, input.shape[0]/dpi), dpi = dpi)
            ax = fig.add_subplot(1,1,1)
            ax.imshow(np.ones(input.shape))
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            #chin
            ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
            #left and right eyebrow
            ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
            #nose
            ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
            #left and right eye
            ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
            #outer and inner lip
            ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
            ax.axis('off')

            fig.canvas.draw()

            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            frame_landmark_list.append((input, data))
            plt.close(fig)
        except:
            print('Error: Video corrupted or no landmarks visible')
    
    for i in range(len(frames_list) - len(frame_landmark_list)):
        # Filling frame_landmark_list in case of error
        frame_landmark_list.append(frame_landmark_list[i])
    
    return frame_landmark_list


def select_images_frames(path_to_images):
    images_list = []
    for image_name in os.listdir(path_to_images):
        img = cv2.imread(os.path.join(path_to_images, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imshow("img",img)
        #cv2.waitKey(0)
        images_list.append(img)
    return images_list


# Compute spectogram
def compute_spectrogram(audio_signal, sample_rate):
    """
    Analyses the source to generate a spectrogram

    :returns:
        # - S: Mel spectrogram
    """
    #sample_rate, audio_signal = wavfile.read(audio_path)
    NFFT = 2048 # 2048
    hoplength = int(NFFT/4) # 512
    pr = 2.0
    f_min = 20
    f_max = 20000
    nmels = 256
    S = melspectrogram(y=audio_signal, sr=sample_rate, n_fft=NFFT, hop_length=hoplength, power=pr,fmin=f_min, fmax=f_max, n_mels=nmels)
    return S


# Get spectrogram for each frame
def get_testing_spectrogram(audio_path, audio_start_frame, fps):
    # Read audio
    audio_signal = AudioSegment.from_file(audio_path)
    audio_signal = audio_signal.set_channels(1)
    frame_audio_length_ms = int(1/fps*1000)

    # Get spectrogram from frame --> Get seconds from audio which correspond to frame
    audio_frame_start_time = audio_start_frame
    chunk = audio_signal[int(audio_frame_start_time*1000):int(audio_frame_start_time*1000)+frame_audio_length_ms]
    audio = chunk.get_array_of_samples()
    samples = np.array(audio).astype(np.float32)
    S = compute_spectrogram(samples, audio_signal.frame_rate)
    fig_spect = plt.figure(figsize=(10, 4))
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=44100, fmax=20000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    im_spect = fig2img(fig_spect)
    plt.close(fig_spect)  
    plt.close('all')

    np_im_spect = cv2.cvtColor(np.array(im_spect)[:,:,:3], cv2.COLOR_RGB2BGR)
    frame_spect_resized = cv2.resize(np_im_spect, (256, 256))
    
    return frame_spect_resized 

# Function that creates camera output, corresponding landmark and spectrogram
def generate_landmarks_and_spectrogram(cap, fps, device, pad, audio_path, fa, frame_count):
    #Get webcam image
    no_pic = True

    while(no_pic == True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            audio_frame_start_time = frame_count/fps

            #Create landmark for face
            frame_landmark_list = []
            spect_list = []

            try:
                input = frame
                preds = fa.get_landmarks(input)[0]
                
                input = crop_and_reshape_img(input, preds, pad=pad)
                preds = crop_and_reshape_preds(preds, pad=pad)
                
                dpi = 100
                fig = plt.figure(figsize=(256/dpi, 256/dpi), dpi = dpi)
                ax = fig.add_subplot(1,1,1)
                ax.imshow(np.ones(input.shape))
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

                #chin
                ax.plot(preds[0:17,0],preds[0:17,1],marker='',markersize=5,linestyle='-',color='green',lw=2)
                #left and right eyebrow
                ax.plot(preds[17:22,0],preds[17:22,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
                ax.plot(preds[22:27,0],preds[22:27,1],marker='',markersize=5,linestyle='-',color='orange',lw=2)
                #nose
                ax.plot(preds[27:31,0],preds[27:31,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
                ax.plot(preds[31:36,0],preds[31:36,1],marker='',markersize=5,linestyle='-',color='blue',lw=2)
                #left and right eye
                ax.plot(preds[36:42,0],preds[36:42,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
                ax.plot(preds[42:48,0],preds[42:48,1],marker='',markersize=5,linestyle='-',color='red',lw=2)
                #outer and inner lip
                ax.plot(preds[48:60,0],preds[48:60,1],marker='',markersize=5,linestyle='-',color='purple',lw=2)
                ax.plot(preds[60:68,0],preds[60:68,1],marker='',markersize=5,linestyle='-',color='pink',lw=2) 
                ax.axis('off')

                fig.canvas.draw()

                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                frame_landmark_list.append((input, data))
                plt.close(fig)
                
                spectrogram = get_testing_spectrogram(audio_path, audio_frame_start_time, fps)
                spect_list.append(spectrogram)

                no_pic = False

            except:
                frame_count+=1
                print('Error: Video corrupted or no landmarks or spectrogram visible')
        else:
            break

    if ret:
        frame_mark = torch.from_numpy(np.array(frame_landmark_list)).type(dtype = torch.float) #K,2,256,256,3
        frame_mark = frame_mark.transpose(2,4).to(device) #K,2,3,256,256
        
        x = frame_mark[0,0].to(device)
        g_y = frame_mark[0,1].to(device)

        g_spect = torch.from_numpy(np.array(spect_list[0])).type(dtype = torch.float) # torch.Size([256, 256, 3])
        g_spect = g_spect.transpose(0,2).to(device) # torch.Size([3, 256, 256])

    else:
        x = g_y = g_spect = None

    return x,g_y,ret, g_spect, frame_count



def select_images_frames(path_to_images):
    images_list = []
    for image_name in os.listdir(path_to_images):
        img = cv2.imread(os.path.join(path_to_images, image_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.imshow("img",img)
        #cv2.waitKey(0)
        images_list.append(img)
    return images_list
    

# This function groups all video frames separated by txt in one list and selects K random ones and saves it to generate spect (FINETUNING)
def select_finetuning_frames(video_path, K):
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if n_frames <= K: # There are not enough frames in the video
        rand_frames_idx = [1]*n_frames
    else:
        rand_frames_idx = [0]*n_frames
        i = 0
        while(i < K):
            idx = random.randint(0, n_frames-1)
            if rand_frames_idx[idx] == 0:
                rand_frames_idx[idx] = 1
                i += 1
    
    frames_list = []
    audio_list = []
    
    # Read until video is completed or no frames needed
    ret = True
    frame_idx = 0
    while(ret and frame_idx < n_frames):
        ret, frame = cap.read()
        
        if ret and rand_frames_idx[frame_idx] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            audio_frame_start_time = frame_idx/fps
            audio_list.append(audio_frame_start_time)
            frames_list.append(frame)
            
        frame_idx += 1

    cap.release()
    return frames_list, audio_list, fps
    