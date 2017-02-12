"""
imRead class is responsible for reading images from a provided path. In addition, the class will automatically convert
images from bgr mode (provided by cv2) to rbg mode (optionally).
"""
import cv2, os
from scipy.misc import toimage
from matplotlib import pyplot as plt
class ImRead(object):
    def __init__(self, bgr2rgd_flag=True):
        assert type(bgr2rgd_flag)==bool,\
            "bgr2rgb_flag has incorrect flag. Bool is expected. {} received.".format(type(type(bgr2rgd_flag)).__name__)
        self.__bgr2rgd_flag = bgr2rgd_flag

    def read(self, path):
        """
        Reads a file from a provided path and optionally converts it into rgb format
        :param path: path to the file
        :return: image
        """
        # 1. Checkt that provided path exists
        assert os.path.exists(path), "read method receivecd incorrect path. The following path doesn't exist {}".\
            format(path)
        # 2. Read the image
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.__bgr2rgd_flag:
            b, g, r = cv2.split(im)
            im = cv2.merge([r, g, b])
        return im


if __name__ == '__main__':
    path = '/Users/aponamaryov/Downloads/coco_train_2014/images/COCO_train2014_000000000659.jpg'
    imRead = ImRead()
    im = imRead.read(path=path)
    plt.imshow(toimage(im))
    print "imRead.read was executed correctly"
