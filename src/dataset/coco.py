"""
COCO class is an adapter for coco dataset that ensures campatibility with ConvDet layer logic
"""
import os
from pycocotools.coco import COCO
from easydict import EasyDict as edict
# Syntax: class(object) create a class inheriting from an object to allow new stype variable management
class coco(object):
    name = None
    @property
    def name(self):
        assert type(self.__name) == str, \
            "Coco dataset name was not set correctly. Coco dataset should be initialized with a name: coco(name='train', path, mc)"""
        return self.__name
    @name.setter
    def name(self, value):
        assert type(value) == str, \
            "Coco dataset name was not set correctly. Coco dataset should be initialized with a name: coco(name='train', path, mc)"""
        self.__name = "coco_" + value

    images_path = "Path to be provided"
    @property
    def images_path(self):
        assert os.path.exists(self.__images_path), "Invalid path: {}".format(self.__images_path)
        return self.__images_path
    @images_path.setter
    def images_path(self, value):
        full_path = os.path.join(self.path, value)
        assert os.path.exists(full_path), "Invalid path: {}".format(full_path)
        self.__images_path = full_path

    annotations_path = "Path to be provided"

    path = "Path to be provided"
    @property
    def path(self):
        assert os.path.exists(
            self.__path), "Invalid path was provided for the data set. The following path doesn't exist: {}"\
            .format(self.__path)
        return self.__path
    @path.setter
    def path(self, path):
        assert type(path) == str, "Invalid path name was provided. Path should be a string."
        assert os.path.exists(path), \
            "Invalid path was provided for the data set. The following path doesn't exist: {}" \
                .format(path)
        self.__path = path

    annotations_file = "File name to be provided"
    @property
    def annotations_file(self):
        assert os.path.exists(self.__annotations_file),\
            "Invalid path was provided for the data set. The following path doesn't exist: {}"\
            .format(self.__annotations_file)
        return self.__annotations_file
    @annotations_file.setter
    def annotations_file(self, value):
        full_path = os.path.join(self.path,
                                 self.annotations_path)
        file_path = os.path.join(full_path, value)
        assert os.path.exists(file_path), "File {} doesn't exis at path {}. Please provide a correct file name at {}" \
            .format(value, file_path, full_path)
        self.__annotations_file = file_path

    def __init__(self, coco_name, coco_path, main_controller):
        """
        COCO class is an adapter for coco dataset that ensures campatibility with ConvDet layer logic.
        The dataset should be initialized with:
        :param name: string with a name of the dataset. Good names can be 'train', 'test', or 'va', but I am not a stickler =)
        :param path: a full path to the folder containing images and annotations folder
        :param mc: main controller containing remaining parameters necessary for the proper initialization. mc should contain:
            - mc.batch_size - an integer greater than 0
            - mc.ANNOTATIONS_FILE_NAME - a file name located in the coco_path/annotations
        """
        self.name = coco_name
        assert type(main_controller.BATCH_SIZE)==int and main_controller.BATCH_SIZE>0, "Incorrect mc.batch_size"
        self.mc = main_controller
        #1. Get an array of image indicies
        self.path = coco_path
        self.images_path = 'images'
        self.annotations_path = 'annotations'
        assert type(mc.ANNOTATIONS_FILE_NAME)==str, "Provide a name of the file containing annotations"
        self.annotations_file = mc.ANNOTATIONS_FILE_NAME
        self.coco = COCO(self.annotations_file)



    def read_batch(self):
        """
        This function reads mc.batch_size images

        :return: image_per_batch, label_per_batch, delta_per_batch, \
        aidx_per_batch, gtbox_per_batch
        """
        mc = self.mc

        for batch_element in xrange(mc.BATCH_SIZE):
            '''
            2. Shuffle the array (if shuffling is enabled)
            3. Read an image
            '''
            print "Step {} executed without errors.".format(batch_element)
        return NotImplementedError

if __name__ == "__main__":
    mc = edict()
    mc.BATCH_SIZE = 10
    mc.ANNOTATIONS_FILE_NAME = 'instances_train2014.json'
    c = coco(coco_name="train", coco_path='/Users/aponamaryov/GitHub/coco', main_controller=mc)
    print c.name
    print c.path
    batch = c.read_batch()
    print batch