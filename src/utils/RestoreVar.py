import tensorflow as tf
import os, sys

class RestoreVar(object):

    # Variables for reporting resuts
    __matched_vars = []
    __different_shape_vars = []
    __missing_from_checkpoint_vars = []

    def __init__(self, checkpoint_path, model_vars, debug=False):
        """
        Class designed to update variables of the session with the values stored in the checkpoint.
        Variables in the session should be initialized.

        The class also provides a convenience method to update variable with values that don't have corresponding values
        :param checkpoint_path:
        :param sess:
        """
        self.path = checkpoint_path
        self.model_vars = model_vars

        try:
            self.checkpoint = tf.train.NewCheckpointReader(self.path)
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        self.__debug = debug

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, value):
        assert os.path.exists(value), "Provided path doesn't exist. Please check the path: {}".format(value)
        self.__path = value

    def update_correspoinding_variables(self):
        """
        Updates graph variables with corresponding values from the checkpoint.
        """

        # Get names of checkpoint variables
        checkpoint_vars_and_shapes = self.checkpoint.get_variable_to_shape_map()

        model_vars = self.model_vars

        # Loop through the graph variables and check if they exist in check point
        for i, mvar in enumerate(model_vars):
            mvar_name = mvar.op.name
            # Check if the shape matches
            if self.checkpoint.has_tensor(mvar_name):
                mvar_shape = mvar.get_shape()
                checkvar_shape = checkpoint_vars_and_shapes[mvar_name]
                if mvar_shape == checkvar_shape:
                    # track matched variables
                    checkpoint_tensor = self.checkpoint.get_tensor(mvar_name)
                    model_vars[i] = mvar.assign(checkpoint_tensor)
                    self.__matched_vars.append(mvar)
                else:
                    # track variables with different shape
                    self.__different_shape_vars.append((mvar, checkvar_shape))
            else:
                # track variables that are not found in the checkpoint
                self.__missing_from_checkpoint_vars.append(mvar)

        if self.__debug:
            self.__report()

    def __report(self):
        print "Update of corresponding variables was executed with the following results:"
        print "{} variables updated successfully:".format(len(self.__matched_vars))
        for var in self.__matched_vars:
            print "   {} with the shape:".format(var.op.name), var.get_shape()
        print ""
        print "{} variables were found in the checkpoint but had different shape:".format(len(self.__different_shape_vars))
        for var in self.__different_shape_vars:
            print "   {} with the shape:".format(var[0].op.name), var[0].get_shape(),\
                "had the following shape in the checkpoint:", var[1]
        print ""
        print "{} variables from the graph weren't fond in the checkpoint:".format(len(self.__missing_from_checkpoint_vars))
        for var in self.__missing_from_checkpoint_vars:
            print "   {} with the shape:".format(var.op.name), var.get_shape()




if __name__ == "__main__":

    def conv(input_var, kernel, output_channels, stride=1, same_padding=True, name="conv"):

        input_channels = input_var.get_shape()[3]
        if not type(input_channels) == int:
            print type(input_channels)
            input_channels = input_channels.value

        with tf.variable_scope(name):
            dev = 1 / float(2 * kernel * kernel * input_channels * output_channels)
            initilizer = tf.truncated_normal(
                shape=[kernel, kernel, input_channels, output_channels], mean=0.0,
                stddev=dev)

            W = tf.Variable(initial_value=initilizer, dtype=tf.float32, name="kernels")

            b = tf.Variable(initial_value=[0.0] * output_channels, dtype=tf.float32, name="biases")

            convolution = tf.nn.conv2d(input_var, W, [1, stride, stride, 1],
                                       padding="SAME" if same_padding else "VALID")

            bias_add = tf.nn.bias_add(convolution, b, name="bias_add")

            activation = tf.nn.relu(bias_add, name="relu")

            return activation, [W, b]


    graph = tf.Graph()

    model = []

    with graph.as_default() as graph:

        input_img = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="input/image_batch")

        conv1, w = conv(input_img, 3, 64, 2, name="conv1")
        model.append(w[0])
        conv2, w = conv(input_img, 3, 16, 2, name="conv2")
        model.append(w[0])
        conv3, w = conv(input_img, 3, 16, 2, name="conv3")
        model.append(w[0])
        conv4, w = conv(input_img, 3, 16, 2, name="conv4")
        model.append(w[0])
        conv5, w = conv(input_img, 3, 16, 2, name="conv5")
        model.append(w[0])

    path = '/Users/aponamaryov/GitHub/TF_SqueezeDet_ObjectDet/logs/squeezeDet1024x1024/train/model.ckpt-0'
    RestoreVariables = RestoreVar(path, model, debug=True)
    RestoreVariables.update_correspoinding_variables()

    with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        tf.global_variables_initializer().run()

        for mvar in model:
            print mvar.op.name + ":", mvar.get_shape()

        extracted_var = model[0]
        print extracted_var.op.name
        print extracted_var.eval()