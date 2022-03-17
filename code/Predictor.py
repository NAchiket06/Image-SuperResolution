import cv2
import numpy
import numpy as np
import os
import ImageModifications as im
import CreateModel as cm
import qualityMetricsCalculation as qm
  
def predict(image_path,weights_path,dataset_path):
    
    # load the srcnn model with weights
    srcnn = cm.model()
    srcnn.load_weights('3051crop_weight_200.h5')
    
    # load the degraded and reference images
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    ref = cv2.imread('images/dataset/{}'.format(file))
    
    # preprocess the image with modcrop
    ref = im.modcrop(ref, 3)
    degraded = im.modcrop(degraded, 3)
    
    # convert the image to YCrCb - (srcnn trained on Y channel)
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)
    
    # create image slice and normalize  
    Y = numpy.zeros((1, temp.shape[0], temp.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255
    
    # perform super-resolution with srcnn
    pre = srcnn.predict(Y, batch_size=1)
    
    # post-process output
    pre *= 255
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    
    # copy Y channel back to image and convert to BGR
    temp = im.shave(temp, 6)
    temp[:, :, 0] = pre[0, :, :, 0]
    output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    
    # remove border from reference and degraged image
    ref = im.shave(ref.astype(np.uint8), 6)
    degraded = im.shave(degraded.astype(np.uint8), 6)
    
    # image quality calculations
    scores=[]
    scores.append(qm.compare_images(degraded,ref))
    scores.append(qm.compare_images(output,ref))

    
    # return images and scores
    return ref, degraded, output, scores

