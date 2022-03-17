import ImageModifications as im
import Predictor as prediction
from matplotlib import pyplot as plt
import qualityMetricsCalculation as quality
import  cv2
import os

datasetPath = 'E:/College/Image_SuperRes/Code/images/dataset'
processedImagePath = 'E:/College/Image_SuperRes/Code/images/processed'

#im.prepare_images('G:/College/Image_SuperRes/Spyder/images/dataset',2,'G:/College/Image_SuperRes/Spyder/images/processed')

trained_weights = '3051crop_weight_200.h5'


def SuperResOneImage():
    ref,degraded, output, scores =prediction.predict('images/processed/001.png', trained_weights, datasetPath)


    print('Degraded Image: \nPSNR: {}\nMSE: {}\n'.format(scores[0][0], scores[0][1]))
    print('Reconstructed Image: \nPSNR: {}\nMSE: {}\n'.format(scores[1][0], scores[1][1]))

    fig,axs = plt.subplots(1,2,figsize=(20,20))
    axs[0].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[1].set_title('SRCNN')

    outputPath = 'E:/College/Image_SuperRes/Code/images/output'
    imgPath = 'G:/College/Image_SuperRes/Code/images/output/temp.png'

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])


    img = cv2.imread(imgPath)
    os.chdir(outputPath)
    filename = 'saveImage.png'
    cv2.imwrite(filename,cv2.cvtColor(output, cv2.COLOR_BGR2RGB))


for file in os.listdir('images/processed'):
    
    # perform super-resolution
    ref, degraded, output, scores = prediction.predict('images/processed/{}'.format(file),trained_weights,datasetPath)
    
    # display images as subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original')
    axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Degraded')
    axs[1].set(xlabel = 'PSNR: {}\nMSE: {} '.format(scores[0][0], scores[0][1]))
    axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    axs[2].set_title('SRCNN')
    axs[2].set(xlabel = 'PSNR: {} \nMSE: {} '.format(scores[1][0], scores[1][1]))

    # remove the x and y ticks
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
      
    print('Saving {}'.format(file))
    fig.savefig('images/output/{}.png'.format(os.path.splitext(file)[0])) 
    plt.close()