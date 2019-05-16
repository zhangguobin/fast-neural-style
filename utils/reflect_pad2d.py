import tensorflow as tf

def reflect_pad2d(x, H0=40, H1=40, W0=40, W1=40):
    'NHWC format is expected for x'
    paddings = tf.constant([[0,0],[H0,H1],[W0,W1],[0,0]])
    return tf.pad(x, paddings, "REFLECT")