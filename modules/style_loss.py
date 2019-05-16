import tensorflow as tf
from modules.gram_matrix import gram_matrix

# see the paper A Neural Algorithm of Artistic Style 
# by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
def style_loss(content_feats, style_grams, style_weights):
    """
    Computes the style loss at a set of layers.
    
    Inputs:
    - feats: list of the features at every layer of the current image.
    - style_targets: List of the same length as feats, where style_targets[i] is
      a Tensor giving the Gram matrix the source style image computed at
      layer style_layers[i].
    - style_weights: List of the same length as style_targets, where style_weights[i]
      is a scalar giving the weight for the style loss at layer style_layers[i].
      
    Returns:
    - style_loss: A Tensor contataining the scalar style loss.
    """
    style_loss = tf.constant(0.0)
    for i in tf.range(len(content_feats)):
        layer_var = gram_matrix(content_feats[i])
        loss_i = tf.reduce_mean((layer_var - style_grams[i])**2) * style_weights[i]
        style_loss = tf.add(style_loss, loss_i)
    return style_loss