import tensorflow as tf
from utils.image_utils import preprocess_dataset
from modules.grad import grad

def train_loop(coco_train, sample_size, xformer, vgg_ext, lr, content_weights, style_weights,
          tv_weight, style_feats, style_grams, style_name):
    logdir = (style_name + 'lr' + str(lr) + 'c' + str(content_weights[0]) +
                's' + str(style_weights[0]) + 'tv' + str(tv_weight))
    print(logdir)
    
    global_step = tf.train.get_or_create_global_step()
    global_step.assign(0)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    
    dataset = coco_train.repeat(2).map(preprocess_dataset).batch(4).prefetch(tf.data.experimental.AUTOTUNE)
    iterator = dataset.make_one_shot_iterator()

    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()
    
    while True:
        try:
            batch_loss_avg = tf.contrib.eager.metrics.Mean()
            x = iterator.get_next()
            loss_value, grads = grad(xformer, vgg_ext, x, style_grams, content_weights,
                                     style_weights, tv_weight)

            optimizer.apply_gradients(zip(grads, xformer.trainable_variables),
                                      global_step)
            batch_loss_avg(loss_value)
            
            # Track progress
            if global_step.numpy() % 100 == 0:
                print("Step {:03d}: Loss: {:.3f}".format(global_step.numpy(), batch_loss_avg.result()))
            with tf.contrib.summary.record_summaries_every_n_global_steps(100):
                batch_loss_avg = tf.contrib.eager.metrics.Mean()
                for i in tf.range(len(grads)):
                    tf.contrib.summary.histogram('grads'+str(i), grads[i])
                    tf.contrib.summary.histogram('weights'+str(i), xformer.trainable_variables[i])
            if global_step.numpy() % 1000 == 0:
                xformer.save_weights(logdir+'/ckp')
        except tf.errors.OutOfRangeError:
            xformer.save_weights(logdir+'/ckp')
            break
