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
import face_alignment

import numpy as np

from matplotlib import pyplot as plt



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


def generate_cropped_landmarks(frames_list, face_aligner, pad=50):
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


def generate_landmarks(cap, device, pad, face_aligner):
    """Input: cap a cv2.VideoCapture object, device the torch.device, 
pad the distance in pixel from border to face

output: x the camera output, g_y the corresponding landmark"""
   
    #Get webcam image
    fa = face_aligner
    no_pic = True
    
    while(no_pic == True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #cv2.imshow("Gen: ", RGB)
            #cv2.waitKey(0)
            frames_list = [RGB]


            #Create landmark for face
            frame_landmark_list = []

            for i in range(len(frames_list)):
                try:
                    input = frames_list[i]
                    #cv2.imshow("input: ", input)
                    #cv2.waitKey(0)
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
                    #plt.show()
                    plt.close(fig)
                    no_pic = False
                except:
                    print('Error: Video corrupted or no landmarks visible')
        else:
            break

    if ret:
    
        frame_mark = torch.from_numpy(np.array(frame_landmark_list)).type(dtype = torch.float) #K,2,256,256,3
        frame_mark = frame_mark.transpose(2,4).to(device) #K,2,3,256,256
        
        x = frame_mark[0,0].to(device)
        g_y = frame_mark[0,1].to(device)

    else:
        x = g_y = None
    
    return x,g_y,ret