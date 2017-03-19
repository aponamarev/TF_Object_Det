# enable import using multiline statements (useful when import many items in one statement)
import os, time, cv2
import tensorflow as tf
import numpy as np
from src.utils.util import bbox_transform
from src.dataset.COCO_Reader.coco import coco as COCO
from src.dataset.CustomQueueRunner import CustomQueueRunner as CQR
from src.config.kitti_squeezeDet_config import kitti_squeezeDet_config as KSC
from src.nets.squeezeDet import SqueezeDet as NET


# setup flags that will be used throughout the algorithm
FLAGS = tf.app.flags.FLAGS

# setup defaults and the description
# define data set flags
tf.app.flags.DEFINE_string('dataset', 'COCO',
                       """This trainier addopted specifically for COCO dataset. Please use other trainers for other datasets.""")
tf.app.flags.DEFINE_string('IMAGES_PATH', 'images/train2014',
                           """Provide a path to where the images are stored.""")
tf.app.flags.DEFINE_string('ANNOTATIONS_FILE_NAME', 'annotations/instances_train2014.json',
                           """Please provide an absolute path to the annotations file.""")


# define network flags
tf.app.flags.DEFINE_string('net', 'squeezeDet',
                       """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path',
                           'logs/train/model.ckpt-0',
                           """Path to the pretrained model.""")

# define training flags
tf.app.flags.DEFINE_string('train_dir', 'logs/train',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")


MC = KSC()
MC.DEBUG = False
MC.IMAGES_PATH = FLAGS.IMAGES_PATH
MC.ANNOTATIONS_FILE_NAME = FLAGS.ANNOTATIONS_FILE_NAME
MC.OUTPUT_RES = (24, 24)
MC.RESIZE_DIM = (768, 768)
MC.BATCH_SIZE = 64
MC.BATCH_CLASSES = ['person', 'car', 'bicycle']

IMDB = COCO(coco_name='train',
            main_controller=MC,
            resize_dim=(MC.RESIZE_DIM[0], MC.RESIZE_DIM[1]))

MC.ANCHOR_BOX = IMDB.ANCHOR_BOX
MC.ANCHORS = len(MC.ANCHOR_BOX)
MC.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
MC.CLASSES = len(IMDB.CLASS_NAMES_AVAILABLE)
MC.IMAGE_WIDTH, MC.IMAGE_HEIGHT = IMDB.resize.dimension_targets

def train():
    """ Executes training procedure that includes:
            1. Create a computational graph, model, database, and a controller
            2. Initialize variables in the model and merge all summaries
            3. Read a minibatch of data
            4. Convert a 2d arrays of inconsistent size (varies based on n of objects) into a list of tuples or tripples
            5. Configure operation that TF should run depending on the step number
            6. Save the model checkpoint periodically.
    """
    graph = tf.Graph()
    with graph.as_default():
        # 1. Create a dataset and a controller
        cqr = CQR(IMDB.get_sample, len(IMDB.ANCHOR_BOX), len(IMDB.CLASS_NAMES_AVAILABLE),
                  img_size=[MC.IMAGE_WIDTH, MC.IMAGE_HEIGHT,3],
                  batch_size=MC.BATCH_SIZE)
        input_dict = {}
        input_dict['image_input'],\
        input_dict['labels'],\
        input_dict['input_mask'],\
        input_dict['box_delta_input'],\
        input_dict['box_input'] = cqr.dequeue

        model = NET(MC, FLAGS.gpu, input_dict)
        # 2. Initialize variables in the model and merge all summaries
        initializer = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph)

        # Launch coordinator that will manage threads
        coord = tf.train.Coordinator()
    graph.finalize()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    sess = tf.Session(config=config, graph=graph)
    print("Beginning training process.")
    sess.run(initializer)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    pass_tracker_start = time.time()
    pass_tracker_prior = pass_tracker_start
    print("Prefetching data. It make take some time...")
    for i in range(cqr.q_capacity):
        cqr.fill_q(sess)
        if i % MC.BATCH_SIZE==0:
            print(".", end="", flush=True)
    print("\nFinished prefetching")

    prior_step = 0

    for step in range(FLAGS.max_steps):

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
                sess.run(op_list, feed_dict={model.keep_prob: MC.KEEP_PROB})

            pass_tracker_end = time.time()

            #viz_summary = sess.run(model.viz_op)
            summary_writer.add_summary(summary_str, step)
            #summary_writer.add_summary(viz_summary, step)

            #Report results
            number_of_steps = step - prior_step
            number_of_steps = number_of_steps if number_of_steps > 0 else 1
            print('\nStep: {}. Timer: {} network passes (with batch size {}): {:.1f} seconds ({:.1f} per batch). Losses: conf_loss: {:.3f}, bbox_loss: {:.3f}, class_loss: {:.3f} and total_loss: {:.3f}'.
                   format(step, number_of_steps, IMDB.mc.BATCH_SIZE,
                          pass_tracker_end - pass_tracker_prior, (pass_tracker_end - pass_tracker_prior)/number_of_steps,
                          conf_loss, bbox_loss, class_loss, loss_value))
            pass_tracker_prior = pass_tracker_end
            prior_step = step

        else:
            _, loss_value, conf_loss, bbox_loss, class_loss = \
                sess.run([model.train_op, model.loss, model.conf_loss, model.bbox_loss, model.class_loss],
                         feed_dict={model.keep_prob: MC.KEEP_PROB})
            print(".", end="", flush=True)

        assert not np.isnan(loss_value), \
            'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
            'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

        # 6. Save the model checkpoint periodically.
        if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
            viz_summary = sess.run(model.viz_op)
            summary_writer.add_summary(summary_str, step)
            summary_writer.add_summary(viz_summary, step)
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path)

        # Prefetch data
        cqr.fill_q_async(sess)

    # Close a queue and cancel all elements in the queue. Request coordinator to stop all the threads.
    sess.run(CQR.queue.close(cancel_pending_enqueues=True))
    coord.request_stop()
    # Tell coordinator to stop any queries to the threads
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    train()
    print("code was executed successfully.")