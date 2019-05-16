import tensorflow as tf

# see "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (Johnson et al., ECCV 2016)

def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    
    It turns out that it's helpful to also encourage smoothness in the image.
    We can do this by adding another term to our loss that penalizes wiggles
    or "total variation" in the pixel values.
    You can compute the "total variation" as the sum of the squares of differences
    in the pixel values for all pairs of pixels that are next to each other (horizontally or vertically).
    
    Inputs:
    - img: Tensor of shape (N, H, W, 3) holding input images
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    
    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    shape = tf.shape(img)
    A = tf.slice(img, [0, 0, 0, 0], shape - [0, 1, 1, 0])
    B = tf.slice(img, [0, 1, 0, 0], shape - [0, 1, 1, 0])
    C = tf.slice(img, [0, 0, 1, 0], shape - [0, 1, 1, 0])
    loss = tv_weight * tf.reduce_sum((A - B)**2 + (A - C)**2) / tf.cast(shape[0], tf.float32)
    return loss