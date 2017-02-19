"""
COCO class is an adapter for coco dataset that ensures campatibility with ConvDet layer logic
"""
import os, cv2
from random import shuffle
from pycocotools.coco import COCO
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

    annotations_file = "File name to be provided"
    @property
    def annotations_file(self):
        assert os.path.exists(self.__annotations_file),\
            "Invalid path was provided for the data set. The following path doesn't exist: {}"\
            .format(self.__annotations_file)
        return self.__annotations_file
    @annotations_file.setter
    def annotations_file(self, value):
        assert os.path.exists(value), "Annotations file doesn't exis at path {}. Please provide a full path to annotatinos." \
            .format(value)
        self.__annotations_file = value

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
            id_array.extend(self.coco.getImgIds(catIds=catIds))

        # update image ids
        self.imgIds = id_array

    def __init__(self, coco_name, main_controller, shuffle=True, resize_dim=(1024, 1024)):
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

        IMDB.__init__(self, resize_dim=resize_dim,
                      feature_map_size=main_controller.OUTPUT_RES,
                      main_controller=main_controller)

        self.name = coco_name
        self.shuffle = shuffle

        #1. Get an array of image indicies
        assert type(main_controller.ANNOTATIONS_FILE_NAME)==str,\
            "Provide a name of the file containing annotations in mc.ANNOTATIONS_FILE_NAME"
        self.annotations_file = main_controller.ANNOTATIONS_FILE_NAME
        self.coco = COCO(self.annotations_file)
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.CLASSES = [category['name'] for category in categories]
        self.CATEGORIES = set([category['supercategory'] for category in categories])
        assert type(main_controller.BATCH_CLASSES) == list and len(main_controller.BATCH_CLASSES)>0,\
            "Provide a list of classes to be learned in this batch through mc.BATCH_CLASSES"
        self.BATCH_CLASSES = main_controller.BATCH_CLASSES

    def provide_img_id(self, id):
        return self.imgIds[id]

    def provide_epoch_size(self):
        return len(self.imgIds)

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
        return [ann['category_id'] for ann in anns]

    def provide_img_gtbboxes(self, id):
        """
        Protocol describing the implementation of a method that provides ground truth bounding boxes
        for the image file based on an image id.
        :param id: dataset specific image id
        :return: an array containing the list of bounding boxes with the following format
        [center_x, center_y, width, height]
        """
        bboxes = []

        # Extract annotation ids
        ann_ids = self.coco.getAnnIds(imgIds=[id],
                                      catIds=self.coco.getCatIds(catNms=self.BATCH_CLASSES)
                                      )
        # get all annotations available
        anns = self.coco.loadAnns(ids=ann_ids)
        # parse annotations into a list
        for ann in anns:
            bbox = self.resize.bboxResize(ann['bbox'])
            bboxes.append([bbox[0] + bbox[2] / 2,
                           bbox[1] + bbox[3] / 2,
                           bbox[2],
                           bbox[3]
                           ])
        return bboxes

    def visualization(self, im, labels=None, bboxes=None):
        text_bound = 3
        fontScale = 0.6
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        if not bboxes == None:
            for idx in xrange(len(bboxes)):
                cx, cy, w, h = [v for v in bboxes[idx]]
                cx1, cy1 = int(cx - w / 2.0), int(cy - h / 2.0)
                cx2, cy2 = int(cx + w / 2.0), int(cy + h / 2.0)
                cv2.rectangle(im, (cx1, cy1), (cx2, cy2), color=256, thickness=1)
                if not labels==None:
                    anns = self.coco.loadCats(ids=labels[idx])[0]
                    txt = "Tag: {}".format(anns['name'])
                    txtSize = cv2.getTextSize(txt, font, fontScale, thickness)[0]
                    cv2.putText(im, txt,
                                (int((cx1+cx2-txtSize[0])/2.0),
                                 cy1-text_bound\
                                     if cy1-txtSize[1]-text_bound>text_bound else cy1+txtSize[1]+text_bound),
                                font, fontScale, 255, thickness=thickness)
        plt.imshow(im)

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from easydict import EasyDict as edict

    mc = edict()
    mc.BATCH_SIZE = 10
    mc.ANNOTATIONS_FILE_NAME = '/Users/aponamaryov/Downloads/coco_train_2014/annotations/instances_train2014.json'
    mc.BATCH_CLASSES = ['person', 'car']
    mc.OUTPUT_RES = (32, 32)
    mc.IMAGES_PATH = '/Users/aponamaryov/Downloads/coco_train_2014/images'
    c = coco(coco_name="train",
             main_controller=mc)
    print "The name of the dataset: {}".format(c.name)
    print "Batch provides images for:  \n", c.BATCH_CLASSES
    image_per_batch,\
    label_per_batch,\
    gtbox_per_batch,\
    aids_per_batch,\
    deltas_per_batch = c.read_batch(5377)
    for id, img in enumerate(image_per_batch):
        c.visualization(img, labels=label_per_batch[id], bboxes=gtbox_per_batch[id])
        """anchors = [c.anchors[v] for v in aids_per_batch[id]]
        c.visualization(img, labels=label_per_batch[id], bboxes=anchors)"""
    print len(c.anchors)
    c.BATCH_CLASSES = ['person', 'dog', 'cat', 'car']
    print c.BATCH_CLASSES
    print len(c.imgIds)