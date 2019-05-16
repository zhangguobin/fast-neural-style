import tensorflow as tf
import tensorflow.keras.backend as K

class InstanceNormalization(tf.keras.layers.Layer):
    def call(self, x):
        mean = K.mean(x, axis=[1,2], keepdims=True)
        norm = K.std(x, axis=[1,2], keepdims=True)
        return (x - mean) / norm