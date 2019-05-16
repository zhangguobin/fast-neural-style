from tensorflow.keras import layers

import sys
sys.path.append('../')
from utils.reflect_pad2d import reflect_pad2d

from modules.InstanceNormalization import InstanceNormalization
from modules.ResnetIdentityBlock import ResnetIdentityBlock

import tensorflow as tf

# Implement the architecture used in jcjohnson/fast-neural-style is
#     c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-3.
# All internal convolutional layers are followed by a ReLU and either batch normalization or instance normalization. 
#     cXsY-Z: A convolutional layer with a kernel size of X, a stride of Y, and Z filters.
#     dX: A downsampling convolutional layer with X filters, 3x3 kernels, and stride 2.
#     RX: A residual block with two convolutional layers and X filters per layer.
#     uX: An upsampling convolutional layer with X filters, 3x3 kernels, and stride 1/2.

# NHWC format is expected
class Transformer(tf.keras.Model):
    def __init__(self, instanceNorm=False, regularizer=None):
        super(Transformer, self).__init__()

        self.instanceNorm = instanceNorm
        # padding due to dimension reduction in following Resnet blocks
        self.reflect_pad = layers.Lambda(reflect_pad2d)
        
        # convolutional layer with a kernel size of 9x9, a stride of 1, and 32 filters
        self.conv1 = layers.Conv2D(filters=32, kernel_size=(9,9), strides=(1, 1),
                                   padding='same',
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer)
        if instanceNorm is False:
            self.conv1_norm = layers.BatchNormalization(beta_regularizer=regularizer,
                                                        gamma_regularizer=regularizer)
        else:
            self.conv1_norm = InstanceNormalization()
        self.conv1_relu = layers.ReLU()

        # downsampling convolutional layer with 64 filters, 3x3 kernels, and stride 2
        self.conv2 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),
                                   padding='same',
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer)
        if instanceNorm is False:
            self.conv2_norm = layers.BatchNormalization(beta_regularizer=regularizer,
                                                        gamma_regularizer=regularizer)
        else:
            self.conv2_norm = InstanceNormalization()
        self.conv2_relu = layers.ReLU()

        # downsampling convolutional layer with 128 filters, 3x3 kernels, and stride 2
        self.conv3 = layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2),
                                   padding='same',
                                   kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer)
        if instanceNorm is False:
            self.conv3_norm = layers.BatchNormalization(beta_regularizer=regularizer,
                                                        gamma_regularizer=regularizer)
        else:
            self.conv3_norm = InstanceNormalization()
        self.conv3_relu = layers.ReLU()

        # residual blocks with two convolutional layers and 128 filters per layer
        self.resnet1 = ResnetIdentityBlock(kernel_size=3, filters=[128,128])
        self.resnet2 = ResnetIdentityBlock(kernel_size=3, filters=[128,128])
        self.resnet3 = ResnetIdentityBlock(kernel_size=3, filters=[128,128])
        self.resnet4 = ResnetIdentityBlock(kernel_size=3, filters=[128,128])
        self.resnet5 = ResnetIdentityBlock(kernel_size=3, filters=[128,128])

        self.conv9 = layers.Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2),
                                            padding='same',
                                            kernel_regularizer=regularizer,
                                            bias_regularizer=regularizer)
        if instanceNorm is False:
            self.conv9_norm = layers.BatchNormalization(beta_regularizer=regularizer,
                                                        gamma_regularizer=regularizer)
        else:
            self.conv9_norm = InstanceNormalization()
        self.conv9_relu = layers.ReLU()

        self.conv10 = layers.Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2),
                                             padding='same',
                                             kernel_regularizer=regularizer,
                                             bias_regularizer=regularizer)
        if instanceNorm is False:
            self.conv10_norm = layers.BatchNormalization(beta_regularizer=regularizer,
                                                         gamma_regularizer=regularizer)
        else:
            self.conv10_norm = InstanceNormalization()
        self.conv10_relu = layers.ReLU()

        self.conv11 = layers.Conv2D(filters=3, kernel_size=(9,9), strides=(1, 1),
                                    padding='same',
                                    kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer)
        self.conv11_tanh = layers.Lambda(tf.keras.backend.tanh)

    def call(self, inputs, training=True):
        x = self.reflect_pad(inputs)
        x = self.conv1(x)
        if self.instanceNorm is False:
            x = self.conv1_norm(x, training=training)
        else:
            x = self.conv1_norm(x)
        x = self.conv1_relu(x)
        
        x = self.conv2(x)
        if self.instanceNorm is False:
            x = self.conv2_norm(x, training=training)
        else:
            x = self.conv2_norm(x)
        x = self.conv2_relu(x)
        
        x = self.conv3(x)
        if self.instanceNorm is False:
            x = self.conv3_norm(x, training=training)
        else:
            x = self.conv3_norm(x)
        x = self.conv3_relu(x)
        
        x = self.resnet1(x)
        x = self.resnet2(x)
        x = self.resnet3(x)
        x = self.resnet4(x)
        x = self.resnet5(x)
        
        x = self.conv9(x)
        if self.instanceNorm is False:
            x = self.conv9_norm(x, training=training)
        else:
            x = self.conv9_norm(x)
        x = self.conv9_relu(x)

        x = self.conv10(x)
        if self.instanceNorm is False:
            x = self.conv10_norm(x, training=training)
        else:
            x = self.conv10_norm(x)
        x = self.conv10_relu(x)
        
        x = self.conv11(x)
        x = self.conv11_tanh(x)
        return x