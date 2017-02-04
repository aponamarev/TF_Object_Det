# enable import using multiline statements (useful when import many items in one statement)
from __future__ import absolute_import
import tensorflow as tf
# import config for kitti
from src.config import kitti_squeezeDet_config
# import dataset class for kitti set
from src.dataset import kitti


# setup flags that will be used throughout the algorithm
FLAGS = tf.app.flags.FLAGS

# setup defaults and the description for flags controlling the data set
tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '/Users/aponamaryov/GitHub/squeezeDet/data/KITTI', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")

# create main controller (mc)
mc = kitti_squeezeDet_config()

# import image dataset (imdb)
imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

if __name__ == '__main__':
    print "code was executed successfully."