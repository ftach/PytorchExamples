""" File which stores configurations and parameter settings used in our project. """

# import the necessary packages
import torch
import os

# All models were trained with the batch size of 6, using the  with initial learning rate of 0.0001 for 20,000 iterations

# DATASET_PATH = os.path.join("dataset", "train")  # base path of the dataset
#
# define the path to the images and masks dataset
# IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
# MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")

TEST_SPLIT = 0.05
VAL_SPLIT = 0.15

# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

MODALITIES = 1
NUM_CLASSES = 4

INIT_LR = 0.0001
NUM_EPOCHS = 2
BATCH_SIZE = 6

# define the input image dimensions
IMG_START_AT = 56  # we will crop the images
IMG_STOP_AT = 184
VOLUME_START_AT = 13
IMG_SIZE = 128  # 128x128x128

BASE_OUTPUT = "output"  # define the path to the base output directory
