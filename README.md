# fast-neural-style

This reproduces work by the paper ["Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (Johnson et al., ECCV 2016)](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).

## The milestones to success:
1. Understand and implement style transfer ["assignment for cs231n"](http://cs231n.github.io/assignments2017/assignment3/)
2. Replace SqueezeNet with ["tf.slim.vgg"](https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py)
3. Replace tf.slim.vgg with ["tf.keras.vgg"](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/vgg19.py)
4. Then implement the tensorflow version of ["fast style transfer"](https://github.com/jcjohnson/fast-neural-style)
5. Got stuck in tunning due to bad setup of style layers {relu1_2, relu2_2, relu3_3, relu4_3}. Thanks to ["OlavHN"] (https://github.com/OlavHN/fast-neural-style), ["lengstrom"](https://github.com/lengstrom/fast-style-transfer) and ["hwalsuklee"](https://github.com/hwalsuklee/tensorflow-fast-style-transfer), reasonable outputs are generated finally.
6. Refactor IPython version to python scripts
