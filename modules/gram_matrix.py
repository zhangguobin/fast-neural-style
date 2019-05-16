import tensorflow as tf

# see the paper A Neural Algorithm of Artistic Style 
# by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.
    The Gram matrix is an approximation to the covariance matrix
    -- we want the activation statistics of our generated image to
    match the activation statistics of our style image, and matching
    the (approximate) covariance is one way to do that. 
    There are a variety of ways you could do this, but the Gram matrix
    is nice because it's easy to compute and in practice shows good results.

    Inputs:
    - features: Tensor of shape (N, H, W, C) giving features for
      N images.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (H * W * C)
        H = tf.getshape(features)[1]
    Returns:
    - gram: Tensor of shape (N, C, C) giving the (optionally normalized)
      Gram matrices for the input images.
    """
    N, H, W, C = tf.shape(features)
    if normalize is True:
        features /= tf.sqrt(tf.cast(H*W*C, features.dtype))
    features = tf.reshape(features, (N,-1,C))
    features_T = tf.transpose(features, (0,2,1))
    grams = tf.matmul(features_T, features)
    return grams