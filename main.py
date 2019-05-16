#!/usr/bin/env python

import tensorflow as tf
tf.enable_eager_execution()

import os
import numpy as np
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from modules.Transformer import Transformer
from modules.gram_matrix import gram_matrix
from train_loop import train_loop
import argparse

parser = argparse.ArgumentParser(description='args for style transfer')
parser.add_argument('--style-image', dest='style_path',
                    help='file path of style image',
                    default='datasets/styles/the_scream.jpg')
parser.add_argument('--learning-rate', dest='lr', type=float,
                    help='learning rate',
                    default=1e-3)
parser.add_argument('--content-weight', dest='content_weight', type=float,
                    help='content weight',
                    default=7.5)
parser.add_argument('--style-weight', dest='style_weight', type=float,
                    help='style weight',
                    default=1e2)
parser.add_argument('--tv-weight', dest='tv_weight', type=float,
                    help='total variance weight',
                    default=2e-4)
args = parser.parse_args()

style_layers = ['block1_conv1', 'block2_conv1',
                'block3_conv1', 'block4_conv1']
content_layers = ['block3_conv2']
content_weights = np.ones(len(content_layers)) * args.content_weight
style_weights = np.ones(len(style_layers)) * args.style_weight
_, style_name = os.path.split(args.style_path)

coco_train = tfds.load(name="coco2014", split=tfds.Split.TRAIN)

vgg_model = VGG16(weights='imagenet', include_top=False)
for layer in vgg_model.layers:
    layer.trainable = False

content_feats = [vgg_model.get_layer(name).output for name in content_layers]
style_feats = [vgg_model.get_layer(name).output for name in style_layers]
vgg_ext = tf.keras.Model(inputs=vgg_model.input,
                         outputs=(content_feats+style_feats))

style_img = image.load_img(args.style_path, target_size=(256, 256))
x = image.img_to_array(style_img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
style_feats = vgg_ext.predict(x)

IDX = len(content_layers)
style_grams = [gram_matrix(layer) for layer in style_feats[IDX:]]

xformer = Transformer(instanceNorm=True)
train_loop(coco_train, 256, xformer, vgg_ext, args.lr, content_weights, style_weights,
           args.tv_weight, style_feats, style_grams, style_name)