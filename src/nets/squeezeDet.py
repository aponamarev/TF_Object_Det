# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""
import joblib
import tensorflow as tf
from src.nets.nn_skeleton import ModelSkeleton

class SqueezeDet(ModelSkeleton):
  def __init__(self, mc, gpu_id, inputs_dict):
    with tf.device('/gpu:{}'.format(gpu_id)):
      ModelSkeleton.__init__(self, mc, inputs_dict) # Initializes inputs

      self._add_forward_graph() # creates a simple network (with fire layers) with the last conv layer stored in
      # self.pred. self.pred has 72 feature maps: 9_boxes * ( 3_classes + 4_deltas + 1_confidence_score )
      self._add_interpretation_graph() # creates class probabilityes, bbox coordinates, and probabilities
      # includes a bunch of summary ops to capture the flow of roi calculation
      self._add_loss_graph()
      self._add_train_graph()
      self._add_viz_graph()

  def _add_forward_graph(self):
    """NN architecture."""

    mc = self.mc
    if mc.LOAD_PRETRAINED_MODEL:
      assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
          'Cannot find pretrained model at the given path:' \
          '  {}'.format(mc.PRETRAINED_MODEL_PATH)
      self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

    conv1 = self._conv_layer(
        'conv1', self.image_input, filters=32, size=3, stride=2,
        padding='SAME', freeze=True)

    pool1 = self._pooling_layer(
        'pool1', conv1, size=3, stride=2, padding='SAME')

    conv2 = self._conv_layer(
        'conv2', pool1, filters=64, size=3, stride=2,
        padding='SAME', freeze=True)

    fire2 = self._fire_layer(
        'fire2', conv2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    fire3 = self._fire_layer(
        'fire3', fire2, s1x1=16, e1x1=64, e3x3=64, freeze=False)
    #combination of convolution with stride of 1 and pooling with stride 2 is more expensive as compared to
    #applying a convolution with stride of 2 directly
    pool3 = self._pooling_layer(
        'pool3', fire3, size=3, stride=2, padding='SAME')

    fire4 = self._fire_layer(
        'fire4', pool3, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    fire5 = self._fire_layer(
        'fire5', fire4, s1x1=32, e1x1=128, e3x3=128, freeze=False)
    pool5 = self._pooling_layer(
        'pool5', fire5, size=3, stride=2, padding='SAME')

    fire6 = self._fire_layer(
        'fire6', pool5, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire7 = self._fire_layer(
        'fire7', fire6, s1x1=48, e1x1=192, e3x3=192, freeze=False)
    fire8 = self._fire_layer(
        'fire8', fire7, s1x1=64, e1x1=256, e3x3=256, freeze=False)
    fire9 = self._fire_layer(
        'fire9', fire8, s1x1=64, e1x1=256, e3x3=256, freeze=False)

    # Two extra fire modules that are not trained before
    fire10 = self._fire_layer(
        'fire10', fire9, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    fire11 = self._fire_layer(
        'fire11', fire10, s1x1=96, e1x1=384, e3x3=384, freeze=False)
    dropout11 = tf.nn.dropout(fire11, self.keep_prob, name='drop11')

    num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
    self.preds = self._conv_layer(
        'conv12', dropout11, filters=num_output, size=3, stride=1,
        padding='SAME', xavier=False, relu=False, stddev=0.0001)

  def _fire_layer(self, layer_name, inputs, s1x1, e1x1, e3x3, freeze=False):
    """Fire layer constructor.

    SqueezeNet is built upon Fire Module, which is comprised of a squeeze layer as input, and two parallel expand layers as output.
    The alternating squeeze and expand layers effectively reduces parameter size without losing too much accuracy.

    Args:
      layer_name: layer name
      inputs: input tensor
      s1x1: number of 1x1 filters in squeeze layer.
      e1x1: number of 1x1 filters in expand layer.
      e3x3: number of 3x3 filters in expand layer.
      freeze: if true, do not train parameters in this layer.
    Returns:
      fire layer operation.
    """

    sq1x1 = self._conv_layer(
        layer_name+'/squeeze1x1', inputs, filters=s1x1, size=1, stride=1,
        padding='SAME', freeze=freeze)
    ex1x1 = self._conv_layer(
        layer_name+'/expand1x1', sq1x1, filters=e1x1, size=1, stride=1,
        padding='SAME', freeze=freeze)
    ex3x3 = self._conv_layer(
        layer_name+'/expand3x3', sq1x1, filters=e3x3, size=3, stride=1,
        padding='SAME', freeze=freeze)

    """ Expansion layers will be concatenated below"""

    return tf.concat([ex1x1, ex3x3], 3, name=layer_name+'/concat')
