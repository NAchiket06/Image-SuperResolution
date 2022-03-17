import tensorflow
from tensorflow import keras
import cv2
import numpy
import matplotlib
import skimage

import PIL
from PIL import Image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
import os


def prepare_images(path, factor):
    
    # loop through the files in the directory
    for file in os.listdir(path):
        
        # open the file
        img = cv2.imread(path + '/' + file)
        
        # find old and new image dimensions
        h, w, _ = img.shape
        #print(h , " ", w)
        new_height = h / factor
        new_width = w / factor
        #print(new_height , " ", new_width)
        # resize the image - down
        img = cv2.resize(img, (100, 100), interpolation = cv2.INTER_LINEAR)
        #img = cv2.resize(img, (h/factor, w/factor))

        # resize the image - up
        img = cv2.resize(img, (384, 384), interpolation = cv2.INTER_LINEAR)
        
        # save the image
        print('Saving {}'.format(file))
        cv2.imwrite('G:/College/Image_SuperRes/Spyder/images/processed/{}'.format(file), img)