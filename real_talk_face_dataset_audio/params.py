#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 11:49 2020

@author: pati
"""

# Necessary configuration parameters for training

# Nmber of frames to load
K = 8

# Number of epochs
epochs = 400

# Batch size
batch_size = 1


# Path to weight
path_to_chkpt = 'model/weights/model_weights.tar' # If not FL

# Path to backup
path_to_backup = 'model/weights/backup_model_weights' # If not FL

#Path to wi weights 
path_to_Wi = 'model/wi_weights/' # If not FL

# Path to training images
training_image_path = 'model/training_images/recent_backup_' # If not FL


# Path to server weight
server_path_to_chkpt = 'server/weights/server_model_weights.tar' 

# Path to server backup
server_path_to_backup = 'server/weights/backup_server_model_weights'

#Path to server wi weights 
server_path_to_Wi = 'server/wi_weights/'

# Path to server training images
server:training_image_path = 'server/training_images/recent_backup_'


# Path to GATV weight
gatv_path_to_chkpt = 'gatv/weights/gatv_model_weights.tar' 

# Path to GATV backup
gatv_path_to_backup = 'gatv/weights/gatv_backup_model_weights'

#Path to GATV wi weights 
gatv_path_to_Wi = 'gatv/wi_weights/'

# Path to training images
gatv_training_image_path = 'gatv/training_images/recent_backup_'


# Path to global weight
global_path_to_chkpt = 'global/weights/global_model_weights.tar'

# Path to global backup
global_path_to_backup = 'global/weights/backup_global_model_weights'

# Path to global training images
global_training_image_path = 'global/training_images/recent_backup_'


