# Python3
import urllib.request, urllib.error, urllib.parse, os, tempfile
import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize

"""
Utility functions used for viewing and processing images.
"""

def blur_image(X):
    """
    A very gentle image blurring operation, to be used as a regularizer for
    image generation.

    Inputs:
    - X: Image data of shape (N, 3, H, W)

    Returns:
    - X_blur: Blurred version of X, of shape (N, 3, H, W)
    """
    from cs231n.fast_layers import conv_forward_fast
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {'stride': 1, 'pad': 1}
    for i in np.arange(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]],
                                  dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]

VGG_MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)

def preprocess_image(images):
    """Preprocess an image for VGG.
    Subtracts the pixel mean.
    """
    images = tf.reverse(images, [-1])
    images = tf.cast(images, tf.float32) - VGG_MEAN_BGR[None,None,None]
    return images

def deprocess_image(images, rescale=False):
    """Undo preprocessing on an image and convert back to uint8."""
    images = images  + VGG_MEAN_BGR[None, None]
    if rescale:
        vmin, vmax = images.min(), images.max()
        images = (images - vmin) / (vmax - vmin)
        images = np.clip(255 * images, 0.0, 255.0).astype(np.uint8)
    else:
        images = np.clip(images, 0.0, 255.0).astype(np.uint8)
    return np.flip(images, 2)

def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, 'wb') as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print('URL Error: ', e.reason, url)
    except urllib.error.HTTPError as e:
        print('HTTP Error: ', e.code, url)


def load_image(filename, size=None):
    """Load and resize an image from disk.

    Inputs:
    - filename: path to file
    - size: size of shortest dimension after rescaling
    """
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img