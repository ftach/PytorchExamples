"""Stores the custom dataset class we will be using for our segmentation dataset. """


import numpy as np
import random
import matplotlib.pyplot as plt
import os
import cv2
import nibabel as nib
from scipy.ndimage import rotate
import tarfile


from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import config

# NOT NEEDED IF NOT THE 1ST TIME
# print("Extracting dataset...")
#
# file = tarfile.open('BraTS2021_Training_Data.tar')
#
# file.extractall('./dataset/')
# file.close()
#
# print("Dataset extracted.")

# cAN BEGIN HERE IF SECOND TIME
print("Loading dataset...")
data_directories = [f.name for f in os.scandir("./dataset/") if f.is_dir()]
# data_directories = [f.path for f in os.scandir("./dataset/") if f.is_dir()]


def pathListIntoIds(dirList):
    x = []
    for i in range(0, len(dirList)):
        x.append(dirList[i][dirList[i].rfind('/')+1:])
    return x


data_ids = pathListIntoIds(data_directories)

# Defining the transformations fo data augmentation
# Define your transformations
transform = transforms.Compose([
    transforms.Lambda(lambda x: np.fliplr(x)),  # Horizontal flip
    transforms.Lambda(lambda x: np.flipud(x)),  # Vertical flip
    transforms.Lambda(lambda x: rotate(x, angle=90)),  # Random rotation
    transforms.Lambda(lambda x: rotate(x, angle=180)),  # Random rotation
    transforms.Lambda(lambda x: rotate(x, angle=270)),  # Random rotation
    transforms.Lambda(lambda x: x * random.uniform(0.9, 1.1)
                      ),  # Random scaling
    transforms.Lambda(lambda x: x + random.uniform(-0.1, 0.1)
                      ),  # Random shifting
])


class CustomDataset(Dataset):
    '''Load the BraTS dataset. Here, we will be using the 4 modalities (Flair, T1w, T1Gd and T2w) of the BraTS data. We also apply 
    data augmentation techniques such as random rotation, random flip, random scaling and random shifting. Then, we crop the images 
    to the desired size (128x128x128) and normalize them. The masks are converted to one-hot encoded tensors. '''

    def __init__(self, list_IDs, img_size=config.IMG_SIZE, dim=(config.IMG_SIZE, config.IMG_SIZE), n_channels=config.MODALITIES, batch_size=config.BATCH_SIZE, training=True, shuffle=True):
        '''Initialization
        Parameters:
            list_IDs (list): List of IDs to load
            dim (tuple): Dimensions of the images (default: (128, 128))
            n_channels (int): Number of channels
            batch_size (int): Batch size
            training (bool): True if training, False if validation/testing
            shuffle (bool): True to shuffle the data every epoch    
        '''
        self.list_IDs = list_IDs
        self.img_size = img_size
        self.dim = dim
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.training = training
        self.shuffle = shuffle

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        # Generate data
        Batch_ids = [self.list_IDs[idx]]
        X, y = self.__data_generation(Batch_ids)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        # Generates data containing batch_size samples
        # X : (batch_size, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size*self.img_size,
                     *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*self.img_size, *self.dim))

        X = np.zeros((self.batch_size*self.img_size,
                     *self.dim, self.n_channels))
        y = np.zeros((self.batch_size*self.img_size,  *self.dim))

        # Generate data
        for c, i in enumerate(Batch_ids):
            # Load NIB files
            case_path = os.path.join("./dataset/", i)
            data_path = os.path.join(case_path, f'{i}_flair.nii.gz')
            flair = nib.load(data_path).get_fdata()

            # data_path = os.path.join(case_path, f'{i}_t1ce.nii.gz')
            # ce = nib.load(data_path).get_fdata()
#
            # data_path = os.path.join(case_path, f'{i}_t1.nii.gz')
            # t1 = nib.load(data_path).get_fdata()
#
            # data_path = os.path.join(case_path, f'{i}_t2.nii.gz')
            # t2 = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'{i}_seg.nii.gz')
            seg = nib.load(data_path).get_fdata()

            # Select the wanted slices and crop the images
            for j in range(self.img_size):

                X[j+(self.img_size*c), :, :, 0] = cv2.resize(flair[config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE,
                                                                   config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE, j+config.VOLUME_START_AT], (config.IMG_SIZE, config.IMG_SIZE))
                # X[j+(self.img_size*c), :, :, 1] = cv2.resize(ce[config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE,
                #                                                config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE, j+config.VOLUME_START_AT], (config.IMG_SIZE, config.IMG_SIZE))
                # X[j+(self.img_size*c), :, :, 2] = cv2.resize(t1[config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE,
                #                                                config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE, j+config.VOLUME_START_AT], (config.IMG_SIZE, config.IMG_SIZE))
                # X[j+(self.img_size*c), :, :, 3] = cv2.resize(t2[config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE,
                #                                                config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE, j+config.VOLUME_START_AT], (config.IMG_SIZE, config.IMG_SIZE))

                y[j+(self.img_size*c), :, :] = cv2.resize(seg[config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE,
                                                              config.IMG_START_AT:config.IMG_START_AT+config.IMG_SIZE, j+config.VOLUME_START_AT], (config.IMG_SIZE, config.IMG_SIZE))
        if self.training:
            # Apply data augmentation
            X = transform(X)
            y = transform(y)

        X = X.reshape(self.batch_size, self.n_channels, self.img_size,
                      *self.dim)
        y = y.reshape(self.batch_size, self.img_size, *self.dim)

        print(X.shape, y.shape)

        # Generate masks
        y[y == 4] = 3  # set to 3 the label 4 (3 is unused)

        # Normalization
        X = X / np.max(X)

        # Convert numpy arrays to PyTorch tensors
        X = torch.from_numpy(X).double()
        y = torch.from_numpy(y).long()

        # shape (IMG_SIZE, IMG_SIZE, VOLUME_SLICES, NUM_CLASSES) for categorical BCE loss
        y = F.one_hot(y)

        return X, y


def train_test_val_split(data_ids, test_size=0.05, val_size=0.15):
    random.shuffle(data_ids)
    test_ids = data_ids[:int(test_size*len(data_ids))]
    train_ids = data_ids[int(test_size*len(data_ids)):]
    val_ids = train_ids[:int(val_size*len(train_ids))]
    return train_ids, test_ids, val_ids
