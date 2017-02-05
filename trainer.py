# enable import using multiline statements (useful when import many items in one statement)
from __future__ import absolute_import
import tensorflow as tf
# import config for kitti
from src.config import kitti_squeezeDet_config
# import dataset class for kitti set
from src.dataset import kitti
# import a net that you'll use
from src.nets import SqueezeDet
# import visualization function
from src.utils.visualization import vis_img
# import conversion from sparse to dense arrays
from src.utils.util import sparse_to_dense, convertToFixedSize


# setup flags that will be used throughout the algorithm
FLAGS = tf.app.flags.FLAGS

# setup defaults and the description
# define data set flags
tf.app.flags.DEFINE_string('dataset', 'KITTI',
                       """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '/Users/aponamaryov/GitHub/squeezeDet/data/KITTI', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                       """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_boolean('augmentation', False, " Define as True if you intend to use data augmentation.")

# define network flags
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                       """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                       """Path to the pretrained model.""")

# define training flags
tf.app.flags.DEFINE_string('train_dir', '/Users/aponamaryov/GitHub/squeezeDet/logs/squeezeDet/train',
                        """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                        """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                        """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                        """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def train():
    """ Executes training procedure that includes:
            1. Check that the provided dataset can be processed
            2. Setup a config and the model for squeezeDet
            3. Initialize a image database
            4. Initialize variables in the model
            5. Read a minibatch of data
            6. Convert a 2d arrays of inconsistent size (varies based on n of objests) into a list of tuples or tripples
    """
    #1. Check that the provided dataset can be processed
    assert FLAGS.dataset in ['KITTI'], \
        'Currently only support KITTI dataset'
    assert FLAGS.net in ['vgg16','resnet50', 'squeezeDet','squeezeDet+'], \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    with tf.Graph().as_default():
        #2. Setup a config and the model for squeezeDet
        if FLAGS.net == 'squeezeDet':
            mc = kitti_squeezeDet_config()
            mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
            model = SqueezeDet(mc, FLAGS.gpu)

        #3. Initilize a image database
        imdb = kitti(data_path=FLAGS.data_path, image_set=FLAGS.image_set, mc=mc)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)):
        #4. Initialize variables in the model
        tf.initialize_all_variables().run()

        for step in xrange(FLAGS.max_steps):
            #5. Read a minibatch of data
            image_per_batch, label_per_batch, box_delta_per_batch, \
            aidx_per_batch, bbox_per_batch = imdb.read_batch()

            # 6. Convert a 2d arrays of inconsistent size (varies based
            # on n of objests) into a list of tuples or tripples
            label_indices, bbox_indices, box_delta_values, mask_indices,\
            box_values = convertToFixedSize(aidx_per_batch=aidx_per_batch,
                                            label_per_batch=label_per_batch,
                                            box_delta_per_batch=box_delta_per_batch,
                                            bbox_per_batch=bbox_per_batch)

            # mask_indices is a list of all anchors responsible for object detection
            # (tuple [image id, anchor id] for each object)



            feed_dict = {
                model.image_input: image_per_batch,
                model.keep_prob: mc.KEEP_PROB,
                model.input_mask: sparse_to_dense(mask_indices, [mc.BATCH_SIZE, mc.ANCHORS])

            }






if __name__ == '__main__':
    train()
    print "code was executed successfully."