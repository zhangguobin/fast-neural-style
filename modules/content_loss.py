import tensorflow as tf

# see the paper A Neural Algorithm of Artistic Style 
# by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
def content_loss(content_weights, content_current, content_original):
    """
    Compute the content loss for style transfer.
    We can generate an image that reflects the content of one image and
    the style of another by incorporating both in our loss function.
    We want to penalize deviations from the content of the content image
    and deviations from the style of the style image. We can then use this
    hybrid loss function to perform gradient descent not on the parameters
    of the model, but instead on the pixel values of our original image.
    
    Inputs:
    - content_weights: scalar constant we multiply the content_loss by.
    - content_current: features of the current image, a list of Tensors.
    - content_target: features of the content image, a list of Tensors, 
        with same shape as content_current.
    
    Returns:
    - scalar content loss
    """
    loss = 0.0
    for i in tf.range(len(content_weights)):
        loss += tf.reduce_mean((content_current[i] - content_original[i])**2) * content_weights[i]
    return loss