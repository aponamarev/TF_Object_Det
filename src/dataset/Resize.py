"""
Resize class is a helper class designed to resize images and annotations for COCO.
Implementation for others classes will be explored later. Current implementation includes the following methods:
    :method: imResize(img) -> image
    :method: bboxResize([x,y,h,w]) - [x_r, y_r, h_r, w_r]

The class is initialized with:
    :param: dimensions - 1d [height, width] array
"""
import cv2, sys
import numpy as np

from scipy import misc
from matplotlib import pyplot as plt

class Resize(object):

    dimension_targets = []
    @property
    def dimension_targets(self):
        return self.__dimension_targets

    @dimension_targets.setter
    def dimension_targets(self, values):
        """
        Ensures the correct set up for dimensions variables
        :param values: 1d [height, width] array
        """
        # ensure tuple type
        values_type = type(values)
        assert values_type==tuple,\
            "dimension_targets received values of incorrect data type. 'tuple' is expected. {} was recieved."\
                .format(values_type.__name__)
        self.__dimension_targets = values

    dimension_received = []
    @property
    def dimension_received(self):
        """
        Ensures that dimesion_recieved read only once. The data will be deleted after it is read.
        """
        # make sure that dimension_reveived is not None
        assert not self.__dimension_received == None,\
            "dimension_recieved has value None. The error caused by reading this variable more than once."
        values = self.__dimension_received
        self.__dimension_received = None
        return values

    @dimension_received.setter
    def dimension_received(self, values):
        """
        Ensures the correct set up for dimensions variables
        :param values: 1d [height, width] array
        """
        # ensure tuple type
        values_type = type(values)
        assert values_type == tuple, \
            "dimensions_received received values of incorrect data type. 'tuple' is expected. {} was recieved." \
                .format(values_type.__name__)
        self.__dimension_received = values

    def __init__(self, dimensinos):
        self.dimension_targets = dimensinos

    def imResize(self, im):
        """
        imResize executes two actions:
            1. setting received_dimensions
            2. resizing the image

        :param im: an image to be resized
        :return: resized image
        """
        # 1. set received_dimensions
        shape = []
        try:
            shape = im.shape[:2]
        except ArithmeticError:
            im = np.array(im)
            shape = im.shape[:2]
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        self.dimension_received = shape

        # 2. resize the image
        im = cv2.resize(im, self.dimension_targets)
        return im


if __name__ == '__main__':
    path = '/Users/aponamaryov/Downloads/coco_train_2014/images/COCO_train2014_000000000659.jpg'

    rsize = Resize(dimensinos=(1000, 500))
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    b,g,r = cv2.split(im)
    im = cv2.merge([r,g,b])
    plt.imshow(misc.toimage(im))
    resized_im = rsize.imResize(im)
    plt.imshow(misc.toimage(resized_im))
    print "resize verification completed successfully!"


