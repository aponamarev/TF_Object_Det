"""
Resize class is a helper class designed to resize images and annotations for COCO.
Implementation for others classes will be explored later. Current implementation includes the following methods:
    :method: imResize(img) -> image
    :method: bboxResize([x,y,w,h]) - [x_r, y_r, w_r, h_r]

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
        return self.__dimension_received

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
            h, w, _ = [float(dim) for dim in im.shape]
            shape = (w,h)
        except ArithmeticError:
            im = np.array(im)
            h, w, _ = [float(dim) for dim in im.shape]
            shape = (w, h)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        self.dimension_received = shape

        # 2. resize the image
        im = cv2.resize(im, self.dimension_targets)
        return im

    def bboxResize(self, gt_bbox):
        """
        Addjust the gt_bbox based on the new dimensions of the image (provided that image was resized first <- otherwise
        error should be raised).

        :param gt_bbox: 1d [x, y, width, height] array

        :return: resized_gt_bbox
        """
        assert type(gt_bbox)==list, "gt_bbox has incorrect datatype. List is expected. {} was received."\
            .format(type(gt_bbox).__name__)

        received_dim = self.dimension_received
        target_dim = self.dimension_targets
        width_adj, height_adj = target_dim[0]/float(received_dim[0]), target_dim[1]/float(received_dim[1])
        gt_bbox[0::2] = [int(gt_bbox[0]*width_adj), int(gt_bbox[2]*width_adj)]
        gt_bbox[1::2] = [int(gt_bbox[1]*height_adj), int(gt_bbox[3]*height_adj)]
        return gt_bbox


if __name__ == '__main__':
    import ImRead
    imread = ImRead()
    path = '/Users/aponamaryov/Downloads/coco_train_2014/images/COCO_train2014_000000000659.jpg'

    rsize = Resize(dimensinos=(1000, 500))
    im = imread.read(path)
    im_ann = im.copy()
    ann = [275, 230, 340, 320]
    im_ann = cv2.rectangle(im_ann,
                               (ann[0]-ann[2]/2, ann[1]-ann[3]/2), (ann[0]+ann[2]/2, ann[1]+ann[3]/2),
                               color=256, thickness=3)
    plt.imshow(misc.toimage(im_ann))
    resized_im = rsize.imResize(im)
    resized_ann = rsize.bboxResize(ann)
    resized_im_ann = cv2.rectangle(resized_im,
                                       (resized_ann[0] - resized_ann[2] / 2, resized_ann[1] - resized_ann[3] / 2),
                                       (resized_ann[0] + resized_ann[2] / 2, resized_ann[1] + resized_ann[3] / 2),
                                       color=256, thickness=2)
    plt.imshow(misc.toimage(resized_im_ann))
    print "resize verification completed successfully!"


