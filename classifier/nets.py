import tensorflow as tf
import params as p

def make_network(input_images, reuse=None):
    with tf.variable_scope("discriminator", reuse=reuse):
        l1 = conv_layer(input_tensor=input_images, size=5, depth=32, name="layer1")  # to 252
        l2 = conv_layer(input_tensor=l1, size=5, depth=32, name="layer2") # to 248
        mp1 = max_pool_2x2(x=l2, name="max_pool_1")  # to 124

        l3 = conv_layer(input_tensor=mp1, size=3, depth=64, name="layer3")  # 122
        l4 = conv_layer(input_tensor=l3, size=3, depth=64, name="layer4")  # 120
        mp2 = max_pool_2x2(x=l4,name="max_pool_2")  # to 60

        l5 = conv_layer(input_tensor=mp2, size=3, depth=128, name="layer5")  # 58
        l6 = conv_layer(input_tensor=l5, size=3, depth=128, name="layer6")  # 56
        mp3 = max_pool_2x2(x=l6,name="max_pool_3")  # to 28

        l7 = conv_layer(input_tensor=mp3, size=3, depth=128, name="layer7")  # 26
        l8 = conv_layer(input_tensor=l7, size=3, depth=128, name="layer8")  # 24
        mp4 = max_pool_2x2(x=l8,name="max_pool_4")  # to 12

        l9 = conv_layer(input_tensor=mp4, size=3, depth=256, name="layer9") # to 10
        mp5 = max_pool_2x2(x=l9,name="max_pool_5")  # to 5

        return FFNN_output(mp5)

def make_generator(input_tensor):
    with tf.variable_scope("generator"):
        tens = conv_layer(input_tensor=input_tensor, size=5, depth=64, name="layer1")
        tens = conv_layer(input_tensor=tens, size=3, depth=64, name="layer2")
        tens = conv_layer(input_tensor=tens, size=3, depth=64, name="layer3")
        tens = conv_layer(input_tensor=tens, size=3, depth=64, name="layer4")
        tens = conv_layer(input_tensor=tens, size=3, depth=3, name="output_rgb")

    return tens + input_tensor



def FFNN_output(input_tensor):
    """Make a full size convolution with the same depth as output classes."""
    with tf.variable_scope("FFNN_output"):
        inc = int(input_tensor.get_shape()[3])
        size_x = int(input_tensor.get_shape()[1])
        size_y = int(input_tensor.get_shape()[2])
        print("size x: %s, size y: %s" % (size_x, size_y))
        depth = p.classes
        w = weight_variable([size_x, size_y, inc, depth], 'w_conv')
        b = bias_variable([depth], 'b_conv')
        c = conv2d_valid(input_tensor, w) + b
        return tf.squeeze(c)


def conv_layer(input_tensor, size, depth, name):
    """
    Create a convolutional layer.

    input_tensor: tensor to convolve upon
    size: kernel size n, symmetric [n x n]
    depth: output channels
    name: node name for tensorboard

    returns: convolved layer.
    """

    with tf.variable_scope(name):
        inc = int(input_tensor.get_shape()[3])
        w = weight_variable([size,size,inc,depth],'w_conv')
        b = bias_variable([depth],'b_conv')
        c = layer_activation(conv2d(input_tensor, w) + b)
        return c

def layer_activation(input_tensor):
    '''
    Convenience function for trying different activations.
    '''
    return tf.nn.relu(input_tensor)

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name=name, initializer=initial)


def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.get_variable(name=name, initializer=initial)


def conv2d(x,W,name=None):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='SAME', name=name)

def conv2d_valid(x,W,name=None):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1], padding='VALID', name=name)

def max_pool_2x2(x, name):
    with tf.name_scope('MP_' + name):
        return tf.nn.max_pool(x,
                          ksize=[1,2,2,1],
                          strides=[1,2,2,1],
                          padding='SAME',
                          name=name)
