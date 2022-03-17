import cv2
import os
import numpy as np
import qualityMetricsCalculation as qm

def prepare_images(path, imageRescaleFactor,processedImageSavePath):
    
    # loop through the files in the directory
    for file in os.listdir(path):
        
        # open the file
        img = cv2.imread(path + '/' + file)
        
        # find old and new image dimensions
        h, w, _ = img.shape

        new_height = h / imageRescaleFactor
        new_width = w / imageRescaleFactor

        # resize the image - down
        imageDimensions = (int(new_width),int(new_height)) 
        img = cv2.resize(img, imageDimensions, interpolation = cv2.INTER_LINEAR) 
        
        # resize the image - up
        imageDimensions = (int(w),int(h))
        img = cv2.resize(img, imageDimensions, interpolation = cv2.INTER_LINEAR)
        
        # save the image
        print('Saving {}'.format(file))
        cv2.imwrite(processedImageSavePath+'/{}'.format(file), img)
        
    
def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 1:sz[1]]
    return img


def shave(image, border):
    img = image[border: -border, border: -border]
    return img

#prepare_images('E:\College\Image_SuperRes\Code\images\dataset',2,'E:\College\Image_SuperRes\Code\images\processed')


def GetQualityMatrixOfPreparedImages():
    director_Path = 'images/dataset'
    for file in os.listdir('images/dataset'):
        
        # open target and reference images
        target = cv2.imread('images/dataset/{}'.format(file))
        ref = cv2.imread('images/processed/{}'.format(file))

        # calculate score
        scores = qm.compare_images(target, ref)

        # print all three scores with new line characters (\n) 
        print('{}\nPSNR: {}\nMSE: {}'.format(file, scores[0], scores[1]))



