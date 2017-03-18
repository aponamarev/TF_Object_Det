import tensorflow as tf
import threading as th

class CustomQueueRunner(object):
    def __init__(self,
                 method_provide_sample,
                 n_boxes,
                 n_classes,
                 img_size=[640, 640, 3],
                 batch_size=10,
                 prefetched=6):
        """
        CustomQueueRunner class is a utility class designed to prefetch and randomize samples.

        The order in which you construct your graph is very important. TensorFlow’s
        sessions support running on separate threads. They do not support construction of the graph.
        This must happen on the main thread. One should construct the CustomRunner on
        the main thread. Its constructor creates the necessary operations on the TensorFlow graph.
        These operations do not need a session at this point.

        After these operations are constructed, initialize all variables on the graph. Only after
        initializing is it okay to start the processing threads. If one of these steps is out of order
        you might encounter race conditions.

        Ref: https://indico.io/blog/tensorflow-data-input-part2-extensions/

        :param method_provide_sample: a hook for a method to be called to generate a new sample
        :param n_boxes:
        :param n_classes:
        :param img_size:
        :param batch_size:
        :param prefetched:
        """
        # Set queue capacity
        self.q_capacity = batch_size * prefetched

        # Assign a method to  be used to generate new sample (singular)
        self.provide_sample = method_provide_sample

        # Define inputs
        self.__inputs = [tf.placeholder(dtype=tf.float32,
                                        shape=img_size,
                                        name="img"),
                         tf.placeholder(dtype=tf.float32,
                                        shape=[n_boxes, n_classes],
                                        name="labels"),
                         tf.placeholder(dtype=tf.float32,
                                        shape=[n_boxes, 1],
                                        name="aids"),
                         tf.placeholder(dtype=tf.float32,
                                        shape=[n_boxes, 4],
                                        name="deltas"),
                         tf.placeholder(dtype=tf.float32,
                                        shape=[n_boxes, 4],
                                        name="bbox_values")]
        # Define the queue and it's ops
        shapes = [v.get_shape().as_list() for v in self.__inputs]
        self.queue = tf.FIFOQueue(capacity=self.q_capacity,
                                  dtypes=[v.dtype for v in self.__inputs],
                                  shapes=shapes)
        self.__q_size = self.queue.size()
        # add summary to observer the state of the prefetching queue
        tf.summary.scalar("prefetching_queue_size", self.__q_size)
        self.dequeue = tf.train.batch(self.queue.dequeue(),batch_size,
                                      num_threads=2,
                                      capacity=6,
                                      shapes=shapes,
                                      name="Batch_{}samples".format(batch_size))
        #self.dequeue = self.queue.dequeue_many(batch_size, name="Batch_{}samples".format(batch_size))
        self.__enqueue_op = self.queue.enqueue(self.__inputs)


    def fill_q(self, sess):
        sess.run(self.__enqueue_op,
                 feed_dict={ps:v for ps,v in zip(self.__inputs, self.provide_sample())})


    def fill_q_async(self, sess):
        """

        The one important thing here is the order in which you construct your graph. TensorFlow’s
        sessions support running on separate threads. They do not support construction of the graph.
        This must happen on the main thread. Once should construct the CustomRunner on
        the main thread. Its constructor creates the necessary operations on the TensorFlow graph.
        These operations do not need a session at this point.

        After these operations are constructed, I initialize all variables on the graph. Only after
        initializing is it okay to start the processing threads. If one of these steps is out of order
        you might encounter race conditions.

        Ref: https://indico.io/blog/tensorflow-data-input-part2-extensions/

        :param sess: that will run a queue on a parallel thread
        :return:
        """
        size = sess.run(self.__q_size)
        for i in range(self.q_capacity - size):
            t = th.Thread(target=self.fill_q, args=[sess])
            t.isDaemon()
            t.start()

if __name__ == "__main__":

    import time
    from easydict import EasyDict as edict
    from COCO_Reader.coco import coco
    from src.config.kitti_squeezeDet_config import kitti_squeezeDet_config

    mc = edict()
    mc.DEBUG = False
    mc.IMAGES_PATH = '../../images'
    mc.ANNOTATIONS_FILE_NAME = '../../annotations/instances_train2014.json'
    mc.OUTPUT_RES = (24, 24)
    mc.RESIZE_DIM = (768, 768)
    mc.BATCH_SIZE = 10
    mc.BATCH_CLASSES = ['person', 'car', 'bicycle']

    imdb = coco("training", mc, resize_dim=(mc.RESIZE_DIM[0],mc.RESIZE_DIM[1]))

    q = CustomQueueRunner(imdb.get_sample,
                          len(imdb.ANCHOR_BOX),
                          len(imdb.CLASS_NAMES_AVAILABLE),
                          [mc.RESIZE_DIM[0], mc.RESIZE_DIM[0],3])

    img, lables, aids, deltas, bbox_values = q.dequeue

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(coord=coord, start=True)

        timer_start = time.time()
        q.fill_q(sess)
        timer_end = time.time()
        print("Filling the queue synchroneously took {:.1f} seconds".\
              format(timer_end-timer_start))

        resulting_img = sess.run(img)
        print("Resulting image has the following shape: {}".\
              format(resulting_img.shape))
