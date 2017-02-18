"""
COCO class is an adapter for coco dataset that ensures campatibility with ConvDet layer logic
"""
import os, cv2
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle
from src.dataset.pycocotools.coco import COCO
from easydict import EasyDict as edict
from imdb_template import imdb_template as IMDB
# Syntax: class(object) create a class inheriting from an object to allow new stype variable management
class coco(IMDB):
    imgIds = []
    @property
    def imgIds(self):
        return self.__imgIds
    @imgIds.setter
    def imgIds(self, values):
        assert type(values)==list,\
            "imgIds is incorrect. Array is expected. {} was recieved.".format(type(values).__name__)
        shuffle_flag = self.shuffle
        if shuffle_flag:
            shuffle(values)

        self.__imgIds = values

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

    BATCH_CLASSES = None
    @property
    def BATCH_CLASSES(self):
        return self.__BATCH_CLASSES
    @BATCH_CLASSES.setter
    def BATCH_CLASSES(self, values):
        for value in values:
            assert value in self.CLASSES,\
                "BATCH_CLASSES array is incorrect. The following class (from the array) " + \
                "is not present in self.CLASSES array: {}.".\
                format(value)
        self.__BATCH_CLASSES = values

        # build an array of file indicies
        id_array = []
        for class_name in self.__BATCH_CLASSES:
            catIds = self.coco.getCatIds(catNms=[class_name])
            ids = self.coco.getImgIds(catIds=catIds)
            id_array.extend(ids)

        # update image ids
        self.imgIds = id_array

    def __init__(self, coco_name, coco_path, main_controller, shuffle=True, resize_dim=(1024, 1024)):
        """
        COCO class is an adapter for coco dataset that ensures campatibility with ConvDet layer logic.
        The dataset should be initialized with:
        :param name: string with a name of the dataset. Good names can be 'train', 'test', or 'va', but I am not a stickler =)
        :param path: a full path to the folder containing images and annotations folder
        :param mc: main controller containing remaining parameters necessary for the proper initialization. mc should contain:
            - mc.batch_size - an integer greater than 0
            - mc.ANNOTATIONS_FILE_NAME - a file name located in the coco_path/annotations
            - mc.BATCH_CLASSES - an array of classes to be learned (at least 1)
        """
        assert not mc.OUTPUT_RES == None,\
            "Please provde the output resolution mc.OUTPUT_RES that describes feature maps size of FCN. Will be used for bbox setup"

        IMDB.__init__(self, resize_dim=resize_dim, feature_map_size=mc.OUTPUT_RES)

        self.name = coco_name
        self.shuffle = shuffle
        assert type(main_controller.BATCH_SIZE)==int and main_controller.BATCH_SIZE>0, "Incorrect mc.batch_size"
        self.mc = main_controller
        #1. Get an array of image indicies
        self.path = coco_path
        self.images_path = 'images'
        self.annotations_path = 'annotations'
        assert type(mc.ANNOTATIONS_FILE_NAME)==str,\
            "Provide a name of the file containing annotations in mc.ANNOTATIONS_FILE_NAME"
        self.annotations_file = mc.ANNOTATIONS_FILE_NAME
        self.coco = COCO(self.annotations_file)
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.CLASSES = [category['name'] for category in categories]
        self.CATEGORIES = set([category['supercategory'] for category in categories])
        assert type(mc.BATCH_CLASSES) == list and len(mc.BATCH_CLASSES)>0,\
            "Provide a list of classes to be learned in this batch through mc.BATCH_CLASSES"
        self.BATCH_CLASSES = mc.BATCH_CLASSES

    def provide_img_file_name(self, id):
        """
        Protocol describing the implementation of a method that provides the name of the image file based on
        an image id.
        :param id: dataset specific image id
        :return: string containing file name
        """
        descriptions = self.coco.loadImgs(id)[0]
        return descriptions['file_name']

    def provide_img_tags(self, id):
        """
        Protocol describing the implementation of a method that provides tags for the image file based on
        an image id.
        :param id: dataset specific image id
        :return: an array containing the list of tags
        """

        # Extract annotation ids
        ann_ids = self.coco.getAnnIds(imgIds=[id],
                                      catIds=self.coco.getCatIds(catNms=self.BATCH_CLASSES)
                                      )
        # get all annotations available
        anns = self.coco.loadAnns(ids=ann_ids)
        # parse annotations into a list
        labels = [ann['category_id'] for ann in anns]

        return labels


    def read_batch(self):
        """
        This function reads mc.batch_size images

        :return: image_per_batch, label_per_batch, delta_per_batch, \
        aidx_per_batch, gtbox_per_batch[cx,cy,w,h]
        """
        image_per_batch,\
        label_per_batch,\
        gtbox_per_batch,\
        aids_per_batch,\
        deltas_per_batch = [],[],[],[],[]
        mc = self.mc

        for batch_element in xrange(mc.BATCH_SIZE):
            '''
            1. Get img_id
            2. Read the file name and annotations
            3. Read and resize an image and annotations
            '''
            #1. Get img_id
            img_id = self.imgIds[batch_element]
            #2. Read the file name
            file_name = self.provide_img_file_name(img_id)


            ann_ids = self.coco.getAnnIds(imgIds=[img_id],
                                          catIds=self.coco.getCatIds(catNms=self.BATCH_CLASSES)
                                          )
            #3. Read and resize an image and annotations
            file_path = os.path.join(self.images_path, file_name)
            try:
                im = self.resize.imResize(self.imread.read(file_path))
                bboxes, category_ids = [],[]
                for id, ann in enumerate(self.coco.loadAnns(ids=ann_ids)):
                    bbox = self.resize.bboxResize(ann['bbox'])
                    bboxes.append([bbox[0] + bbox[2]/2,
                                   bbox[1] + bbox[3]/2,
                                   bbox[2],
                                   bbox[3]
                                   ])
                image_per_batch.append(im)
                label_per_batch.append(self.provide_img_tags(img_id))
                gtbox_per_batch.append(bboxes)
                # provide anchor ids for each image
                aids = self.find_anchor_ids(bboxes)
                # add image anchors to the batch
                aids_per_batch.append(aids)
                # calculate deltas for each anchor and add them to the delta_per_batch
                deltas_per_batch.append(self.estimate_deltas(bboxes, aids))
            except:
                pass

        return image_per_batch, label_per_batch, gtbox_per_batch, aids_per_batch, deltas_per_batch
    def visualization(self, im, labels=None, bboxes=None):
        text_bound = 3
        fontScale = 0.4
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        if not bboxes == None:
            for idx in xrange(len(bboxes)):
                cx, cy, w, h = [v for v in bboxes[idx]]
                cx1, cy1 = int(cx - w / 2.0), int(cy - h / 2.0)
                cx2, cy2 = int(cx + w / 2.0), int(cy + h / 2.0)
                cv2.rectangle(im, (cx1, cy1), (cx2, cy2), color=256, thickness=1)
                if not labels==None:
                    txt = "Tag: {}".format(self.CLASSES[labels[idx]])
                    txtSize = cv2.getTextSize(txt, font, fontScale, thickness)[0]
                    cv2.putText(im, txt,
                                (int((cx1+cx2-txtSize[0])/2.0),
                                 cy1-text_bound\
                                     if cy1-txtSize[1]-text_bound>text_bound else cy1+txtSize[1]+text_bound),
                                font, fontScale, 255, thickness=thickness)
        plt.imshow(im)

if __name__ == "__main__":
    mc = edict()
    mc.BATCH_SIZE = 10
    mc.ANNOTATIONS_FILE_NAME = 'instances_train2014.json'
    mc.BATCH_CLASSES = ['person', 'car']
    mc.OUTPUT_RES = (32,32)
    c = coco(coco_name="train",
             coco_path='/Users/aponamaryov/GitHub/coco',
             main_controller=mc)
    print "The name of the dataset: {}".format(c.name)
    print "The dataset is located at: {}".format(c.path)
    print "Batch provides images for:  \n", c.BATCH_CLASSES
    image_per_batch,\
    label_per_batch,\
    gtbox_per_batch,\
    aids_per_batch, \
    deltas_per_batch = c.read_batch()
    for id, img in enumerate(image_per_batch):
        c.visualization(img, labels=label_per_batch[id], bboxes=gtbox_per_batch[id])
        """anchors = [c.anchors[v] for v in aids_per_batch[id]]
        c.visualization(img, labels=label_per_batch[id], bboxes=anchors)"""
    print len(c.anchors)
    c.BATCH_CLASSES = ['person', 'dog', 'cat', 'car']
    print c.BATCH_CLASSES
    batch = c.read_batch()
    print len(c.imgIds)
    print batch