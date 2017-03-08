# enable import using multiline statements (useful when import many items in one statement)
from __future__ import absolute_import, print_function
import cv2, os, time, threading
import tensorflow as tf
import numpy as np
from src.dataset.COCO_Reader import coco
# import conversion from sparse to dense arrays
from src.utils.util import sparse_to_dense, convertToFixedSize, bbox_transform, bgr_to_rgb


# setup flags that will be used throughout the algorithm
FLAGS = tf.app.flags.FLAGS

# setup defaults and the description
# define data set flags
tf.app.flags.DEFINE_string('dataset', 'COCO',
                       """This trainier addopted specifically for COCO dataset. Please use other trainers for other datasets.""")
tf.app.flags.DEFINE_string('IMAGES_PATH', '/Users/aponamaryov/Downloads/coco_train_2014/images',
                           """Provide a path to where the images are stored.""")
tf.app.flags.DEFINE_string('ANNOTATIONS_FILE_NAME', '/Users/aponamaryov/Downloads/coco_train_2014/annotations/instances_train2014.json',
                           """Please provide an absolute path to the annotations file.""")


# define network flags
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                       """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path',
                           '/Users/aponamaryov/GitHub/TF_SqueezeDet_ObjectDet/logs/squeezeDet1024x1024/train/model.ckpt-0',
                       """Path to the pretrained model.""")

# define training flags
tf.app.flags.DEFINE_string('train_dir', 'logs/squeezeDet1024x1024/train',
                        """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                        """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                        """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                        """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


def defineComputGraph(FLAGS, computational_graph):
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

    # 1. Check that the provided dataset can be processed
    assert FLAGS.dataset in ['KITTI','COCO'], \
        'Currently only support KITTI and COCO datasets'
    assert FLAGS.net in ['vgg16', 'resnet50', 'squeezeDet', 'squeezeDet+'], \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    with computational_graph.as_default():

        # 3. Initilize a image database
        if FLAGS.dataset == 'KITTI':
            mc = kitti_squeezeDet_config()
            imdb = kitti(data_path=FLAGS.data_path, image_set=FLAGS.image_set, mc=mc)
        if FLAGS.dataset == 'COCO':
            mc = kitti_squeezeDet_config()
            mc.DEBUG = False
            mc.IMAGES_PATH = FLAGS.IMAGES_PATH
            mc.ANNOTATIONS_FILE_NAME = FLAGS.ANNOTATIONS_FILE_NAME
            mc.OUTPUT_RES = (24, 24)
            mc.BATCH_SIZE = 10
            mc.BATCH_CLASSES = ['person', 'car', 'bicycle']
            # Dimensions for:
            # imgs, bbox_deltas, masks, dense_labels, bbox_values
            mc.OUTPUT_SHAPES = [[768, 768, 3],
                                [mc.OUTPUT_RES[0] * mc.OUTPUT_RES[1] * 9, 4],
                                [mc.OUTPUT_RES[0] * mc.OUTPUT_RES[1] * 9, 1],
                                [mc.OUTPUT_RES[0] * mc.OUTPUT_RES[1] * 9, 3],
                                [mc.OUTPUT_RES[0] * mc.OUTPUT_RES[1] * 9, 4]]
            mc.OUTPUT_DTYPES = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            imdb = coco(coco_name='train',
                        main_controller=mc,
                        resize_dim=(mc.OUTPUT_SHAPES[0][0], mc.OUTPUT_SHAPES[0][1]))

        # 2. Setup a config and the model for squeezeDet
        if FLAGS.net == 'squeezeDet':
            mc.ANCHOR_BOX = imdb.ANCHOR_BOX
            mc.ANCHORS = len(mc.ANCHOR_BOX)
            mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
            mc.CLASSES = 3
            mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT = imdb.resize.dimension_targets

            inputs_dict = {'image_input':[], 'input_mask':[], 'box_delta_input':[], 'box_input':[], 'labels':[]}

            inputs_dict['image_input'], inputs_dict['box_delta_input'], inputs_dict['input_mask'], \
            inputs_dict['labels'], inputs_dict['box_input'] = imdb.get_batch

            model = SqueezeDet(mc, FLAGS.gpu, inputs_dict)

        return model, imdb, mc

def train():
    """ Executes training procedure that includes:
            1. Create a computational graph, model, database, and a controller
            2. Initialize variables in the model and merge all summaries
            3. Read a minibatch of data
            4. Convert a 2d arrays of inconsistent size (varies based on n of objects) into a list of tuples or tripples
            5. Configure operation that TF should run depending on the step number
            6. Save the model checkpoint periodically.
    """

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    with sess:
        # 1. Create a computational graph, model, database, and a controller
        model, imdb, mc = defineComputGraph(FLAGS, computational_graph=sess.graph)
        print("Beginning training process.")
        # 2. Initialize variables in the model and merge all summaries
        # old version - tf.initialize_all_variables().run()
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        # old version - saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.Saver(tf.global_variables())

        summary_op = tf.summary.merge_all()

        # Prefetch data
        # Enqueue one batch sequentially to ensure that there is at least one batch in the pipeline
        print("... filling in the data pipeline with minibatches. It may take some time.")
        imdb.fill_queue(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        # Launch coordinator that will manage threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        feed_dict = {model.keep_prob: mc.KEEP_PROB}

        pass_tracker_start = time.time()
        pass_tracker_prior = pass_tracker_start

        prior_step = 0

        for step in xrange(FLAGS.max_steps):

            #print("Number of elements in the queue: {}".format(imdb.queue.size().eval()))

            imdb.fill_queue(sess)

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

                _, loss_value, summary_str, det_boxes, det_probs, det_class, conf_loss, bbox_loss, class_loss = \
                    sess.run(op_list, feed_dict=feed_dict)

                pass_tracker_end = time.time()

                viz_summary = sess.run(model.viz_op, feed_dict={model.image_to_show: model.image_input.eval()})
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(viz_summary, step)

                #Report results
                number_of_steps = step - prior_step
                number_of_steps = number_of_steps if number_of_steps > 0 else 1
                print('\nStep: {}. Timer: {} network passes (with batch size {}): {:.1f} seconds ({:.1f} per batch). Losses: conf_loss: {:.3f}, bbox_loss: {:.3f}, class_loss: {:.3f} and total_loss: {:.3f}'.
                       format(step, number_of_steps, imdb.mc.BATCH_SIZE,
                              pass_tracker_end - pass_tracker_prior, (pass_tracker_end - pass_tracker_prior)/number_of_steps,
                              conf_loss, bbox_loss, class_loss, loss_value))
                pass_tracker_prior = pass_tracker_end
                prior_step = step

            else:
                _, loss_value, conf_loss, bbox_loss, class_loss = \
                    sess.run([model.train_op, model.loss, model.conf_loss, model.bbox_loss, model.class_loss],
                             feed_dict=feed_dict)
                print(".", end="")

            assert not np.isnan(loss_value), \
                'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
                'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

            # 6. Save the model checkpoint periodically.
            if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
                viz_summary = sess.run(model.viz_op, feed_dict={model.image_to_show: model.image_input.eval()})
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(viz_summary, step)
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        # Close a queue and cancel all elements in the queue. Request coordinator to stop all the threads.
        sess.run(imdb.queue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        # Tell coordinator to stop any queries to the threads
        coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()
    print("code was executed successfully.")