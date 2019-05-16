import tensorflow as tf
from modules.total_loss import total_loss

def grad(xformer, vgg_ext, orig_imgs, style_grams, content_weights, style_weights, tv_weight):
    with tf.GradientTape() as tape:
        loss_value = total_loss(xformer, vgg_ext, orig_imgs, style_grams, content_weights, style_weights, tv_weight)
    return loss_value, tape.gradient(loss_value, xformer.trainable_variables)