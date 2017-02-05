# Author: Alexander Ponamarev (alex.ponamaryov@gmail.com)
import cv2
import numpy as np
from scipy.misc import toimage
from matplotlib import pyplot as plt

def vis_img(img, dataset=None, means=None):
    if dataset:
        assert dataset in ['KITTI','VOC'], "Error: the dataset should be KITTI or VOC"
    if dataset=='KITTI':
        if not means==None:
            img += means

    __vis(img)

def __vis(img):
    img = np.array(img).astype(np.float)
    shape = img.shape
    plt.imshow(toimage(img))