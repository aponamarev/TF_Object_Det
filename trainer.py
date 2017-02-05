# enable import using multiline statements (useful when import many items in one statement)
from __future__ import absolute_import
import os
import tensorflow as tf
import numpy as np
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

def defineComputGraph(FLAGS):
    """
    1. Check that the provided dataset can be processed
    2. Setup a config and the model for squeezeDet
    3. Initialize a image database

    :param FLAGS: Should contain    FLAGS.pretrained_model_path,
                                    FLAGS.data_path
                                    FLAGS.image_set
                                    FLAGS.pretrained_model_path
                                    FLAGS.gpu

    :return: model and the database
    """
    # import config for kitti
    from src.config import kitti_squeezeDet_config
    # import dataset class for kitti set
    from src.dataset import kitti
    # import a net that you'll use
    from src.nets import SqueezeDet
    # import visualization function

    # 1. Check that the provided dataset can be processed
    assert FLAGS.dataset in ['KITTI'], \
        'Currently only support KITTI dataset'
    assert FLAGS.net in ['vgg16', 'resnet50', 'squeezeDet', 'squeezeDet+'], \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    graph = tf.Graph()
    with graph.as_default():

        # 2. Setup a config and the model for squeezeDet
        if FLAGS.net == 'squeezeDet':
            mc = kitti_squeezeDet_config()
            mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
            model = SqueezeDet(mc, FLAGS.gpu)

        # 3. Initilize a image database
        imdb = kitti(data_path=FLAGS.data_path, image_set=FLAGS.image_set, mc=mc)
        return graph, model, imdb, mc

def train():
    """ Executes training procedure that includes:
            1. Create a computational graph, model, database, and a controller
            2. Initialize variables in the model and merge all summaries
            3. Read a minibatch of data
            4. Convert a 2d arrays of inconsistent size (varies based on n of objests) into a list of tuples or tripples
            5. Configure operation that TF should run depending on the step number
            6. Save the model checkpoint periodically.
    """

    # 1. Create a computational graph, model, database, and a controller
    graph, model, imdb, mc = defineComputGraph(FLAGS)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=graph) as sess:

        # 2. Initialize variables in the model and merge all summaries
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        summary_op = tf.merge_all_summaries()

        tf.train.start_queue_runners(sess=sess)

        for step in xrange(FLAGS.max_steps):
            # 3. Read a minibatch of data
            image_per_batch, label_per_batch, box_delta_per_batch, \
            aidx_per_batch, bbox_per_batch = imdb.read_batch()

            # 4. Convert a 2d arrays of inconsistent size (varies based
            # on n of objests) into a list of tuples or tripples
            label_indices, \
            bbox_indices, \
            box_delta_values, \
            mask_indices, \
            box_values = convertToFixedSize(
                aidx_per_batch=aidx_per_batch,
                label_per_batch=label_per_batch,
                box_delta_per_batch=box_delta_per_batch,
                bbox_per_batch=bbox_per_batch
            )

            feed_dict = {
                model.keep_prob: mc.KEEP_PROB,

                model.image_input: image_per_batch,

                model.input_mask: np.reshape(
                    sparse_to_dense(mask_indices,[mc.BATCH_SIZE, mc.ANCHORS],
                                    np.ones(len(mask_indices), dtype=np.float)),
                    [mc.BATCH_SIZE, mc.ANCHORS, 1]),

                model.box_delta_input: sparse_to_dense(
                    bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
                    box_delta_values),

                model.box_input: sparse_to_dense(
                    bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, 4],
                    box_values),

                model.labels: sparse_to_dense(
                    label_indices, [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
                    np.ones(len(label_indices), dtype=np.float))
            }

            # 5. Configure operation that TF should run depending on the step number
            if step % FLAGS.summary_step == 0:
                op_list = [
                    model.train_op,

                    model.loss,
                    summary_op,
                    model.det_boxes,
                    model.det_probs,
                    model.det_class,
                    model.conf_loss,
                    model.bbox_loss,
                    model.class_loss
                ]

                _, \
                loss_value, \
                summary_str, \
                det_boxes, \
                det_probs, \
                det_class, \
                conf_loss, \
                bbox_loss, \
                class_loss = sess.run(op_list, feed_dict=feed_dict)

            else:
                _, \
                loss_value, \
                conf_loss, \
                bbox_loss, \
                class_loss = sess.run(
                    [model.train_op,
                     model.loss,
                     model.conf_loss,
                     model.bbox_loss,
                     model.class_loss], feed_dict=feed_dict)

            assert not np.isnan(loss_value), \
                'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
                'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

            # 6. Save the model checkpoint periodically.
            if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
    print "code was executed successfully."