# enable import using multiline statements (useful when import many items in one statement)
from __future__ import absolute_import
import cv2
import os
import tensorflow as tf
import numpy as np
from src.dataset import coco
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
                           '/Users/aponamaryov/GitHub/TF_SqueezeDet_ObjectDet/data/model_checkpoints/squeezeDet/model.ckpt-87000',
                       """Path to the pretrained model.""")

# define training flags
tf.app.flags.DEFINE_string('train_dir', './logs/squeezeDet/train',
                        """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                        """Maximum number of batches to run.""")
tf.app.flags.DEFINE_integer('summary_step', 10,
                        """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 1000,
                        """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def _draw_box(im, box_list, label_list, color=(0,255,0), cdict=None, form='center'):
    assert form == 'center' or form == 'diagonal', 'bounding box format not accepted: {}.'.format(form)

    for bbox, label in zip(box_list, label_list):

        if form == 'center':
            bbox = bbox_transform(bbox)

        xmin, ymin, xmax, ymax = [int(b) for b in bbox]

        l = label.split(':')[0] # text before "CLASS: (PROB)"
        if cdict and l in cdict:
            c = cdict[l]
        else:
            c = color

        # draw box
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 1)
        # draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, label, (xmin, ymax), font, 0.3, c, 1)

def _viz_prediction_result(model, images, bboxes, labels, batch_det_bbox,
                           batch_det_class, batch_det_prob):
    mc = model.mc

    for i in range(len(images)):
        # draw ground truth
        _draw_box(images[i], bboxes[i], [mc.CLASS_NAMES[idx] for idx in labels[i]], (0, 255, 0))

        # draw prediction
        det_bbox, det_prob, det_class = model.filter_prediction(
            batch_det_bbox[i], batch_det_prob[i], batch_det_class[i])

        keep_idx    = [idx for idx in range(len(det_prob)) \
                          if det_prob[idx] > mc.PLOT_PROB_THRESH]
        det_bbox    = [det_bbox[idx] for idx in keep_idx]
        det_prob    = [det_prob[idx] for idx in keep_idx]
        det_class   = [det_class[idx] for idx in keep_idx]

        _draw_box(
            images[i], det_bbox,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(det_class, det_prob)],
            (0, 0, 255))

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
    assert FLAGS.dataset in ['KITTI','COCO'], \
        'Currently only support KITTI and COCO datasets'
    assert FLAGS.net in ['vgg16', 'resnet50', 'squeezeDet', 'squeezeDet+'], \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)

    graph = tf.Graph()
    with graph.as_default():

        # 3. Initilize a image database
        if FLAGS.dataset == 'KITTI':
            imdb = kitti(data_path=FLAGS.data_path, image_set=FLAGS.image_set, mc=mc)
        if FLAGS.dataset == 'COCO':
            mc = kitti_squeezeDet_config()
            mc.IMAGES_PATH = FLAGS.IMAGES_PATH
            mc.ANNOTATIONS_FILE_NAME = FLAGS.ANNOTATIONS_FILE_NAME
            mc.OUTPUT_RES = (63, 63)
            mc.BATCH_SIZE = 10
            mc.BATCH_CLASSES = ('person', 'car', 'bicycle')
            imdb = coco(coco_name='train',
                        main_controller=mc,
                        resize_dim=(1024, 1024))

        # 2. Setup a config and the model for squeezeDet
        if FLAGS.net == 'squeezeDet':
            mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
            mc.CLASSES = 3
            mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT = imdb.resize.dimension_targets
            model = SqueezeDet(mc, FLAGS.gpu)

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
        #saver.restore(sess, FLAGS.pretrained_model_path)

        summary_op = tf.merge_all_summaries()

        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            # 3. Read a minibatch of data
            image_per_batch,\
            label_per_batch,\
            gtbbox_per_batch,\
            aidx_per_batch,\
            box_delta_per_batch= imdb.read_batch(step=step)

            label_per_batch = [imdb.tranform_cocoID2batchID(v) for v in label_per_batch]

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
                bbox_per_batch=gtbbox_per_batch
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

                _viz_prediction_result(
                    model, image_per_batch, gtbbox_per_batch, label_per_batch, det_boxes,
                    det_class, det_probs)
                image_per_batch = bgr_to_rgb(image_per_batch)
                viz_summary = sess.run(
                    model.viz_op, feed_dict={model.image_to_show: image_per_batch})
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(viz_summary, step)

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
                viz_summary = sess.run(model.viz_op, feed_dict={model.image_to_show: image_per_batch})
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(viz_summary, step)
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    train()
    print "code was executed successfully."