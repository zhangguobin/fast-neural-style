from tensorflow.keras import layers
from modules.InstanceNormalization import InstanceNormalization

import tensorflow as tf

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, instanceNorm=False, regularizer=None):
        super(ResnetIdentityBlock, self).__init__()
        filters1, filters2 = filters
        self.instanceNorm = instanceNorm
        self.conv2a = layers.Conv2D(filters1, kernel_size,
                                    padding='valid',
                                    kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)
        if instanceNorm is False:
            self.norm2a = layers.BatchNormalization(beta_regularizer=regularizer,
                                                    gamma_regularizer=regularizer)
        else:
            self.norm2a = InstanceNormalization()
            
        self.relu2a = layers.ReLU()

        self.conv2b = layers.Conv2D(filters2, kernel_size,
                                    padding='valid',
                                    kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)
        if instanceNorm is False:
            self.norm2b = layers.BatchNormalization(beta_regularizer=regularizer,
                                                    gamma_regularizer=regularizer)
        else:
            self.norm2b = InstanceNormalization()
            
        self.crop2d = layers.Cropping2D(cropping=2)

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        if self.instanceNorm is False:
            self.norm2a(x, training=training)
        else:
            self.norm2a(x)
        x = self.relu2a(x)

        x = self.conv2b(x)
        if self.instanceNorm is False:
            self.norm2b(x, training=training)
        else:
            self.norm2b(x)

        # center crop due to no padding in Conv2D
        input_cropped = self.crop2d(input_tensor)

        return (x + input_cropped)