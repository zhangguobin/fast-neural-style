import tensorflow as tf

# see "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (Johnson et al., ECCV 2016)

import sys
sys.path.append('../')
from utils.image_utils import preprocess_image
from modules.content_loss import content_loss
from modules.style_loss import style_loss
from modules.tv_loss import tv_loss

VGG_MEAN_BGR = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)

def total_loss(xformer, vgg_x, orig_imgs, style_grams, content_weights, style_weights, tv_weight):
    xformer_out = xformer(orig_imgs)
    gen_img = (xformer_out + 1) / 2 * 256
    gen_normed = gen_img - VGG_MEAN_BGR[None,None,None]
    
    orig_feats = vgg_x(orig_imgs)
    gen_feats = vgg_x(gen_normed)
    
    IDX = len(content_weights)
    c_loss = content_loss(content_weights, gen_feats[:IDX], orig_feats[:IDX]) 
    s_loss = style_loss(gen_feats[IDX:], style_grams, style_weights)
    t_loss = tv_loss(gen_normed, tv_weight)
    
    with tf.contrib.summary.record_summaries_every_n_global_steps(100):
        tf.contrib.summary.scalar('content_loss', c_loss)
        tf.contrib.summary.scalar('style_loss', s_loss)
        tf.contrib.summary.scalar('tv_loss', t_loss)
        for i in tf.range(len(xformer_out)):
            tf.contrib.summary.histogram('xform_output' + str(i), xformer_out[i])
        for i in tf.range(len(orig_feats)):
            tf.contrib.summary.histogram('orig_feats' + str(i), orig_feats[i])
        for i in tf.range(len(gen_feats)):
            tf.contrib.summary.histogram('gen_feats' + str(i), gen_feats[i])
        tf.contrib.summary.image('samples', gen_img[...,::-1], max_images=4)
    return c_loss + s_loss + t_loss