import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib import losses
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import ops

slim = tf.contrib.slim

log_dir = "log" # Where to save logs and checkpoints to
batch_size = 64
max_iterations = 100000000
sum_per = 5 # Create a summary every this many steps
save_per = 1000 # Save every this many steps
learning_rate = 0.00005
d_iters = 5 # Number of discriminator training steps per generator training step
z_dim = 100 # Dimension of the noise vector
c = 0.01 # Value to which to clip the discriminator weights
clip_per = 1 # Experimental. Clip discriminator weights every this many steps. Only works reliably if clip_per=<d_iters

# Create log dir if it doesn't already exist
if not os.path.exists(log_dir):
    os.mkdir(log_dir)


def generator(z, training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    z = tf.reshape(z, [batch_size, 1, 1, z_dim])

    with arg_scope(
        [slim.conv2d_transpose],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params,
        scope="transposed_convolution"):

	# Each tuple is (number of channels, kernel size, stride)
        l = [(1024, [3,3], [2,2]), (512, [3,3], [2,2]), (256, [3,3], [2,2]), (128, [3,3], [2,2]),
            (128, [3,3], [2,2]), (64, [3,3], [2,2]), (3, [3,3], [2,2])]
        gen = slim.stack(z, slim.conv2d_transpose, l)

    gen = tf.tanh(gen)

    return gen


def discriminator(img, training=True, weight_decay=0.0001, batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
        'is_training': training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': ops.GraphKeys.UPDATE_OPS,
    }

    with arg_scope(
        [slim.conv2d],
        weights_regularizer=regularizers.l2_regularizer(weight_decay),
        weights_initializer=initializers.variance_scaling_initializer(),
        activation_fn=nn_ops.relu,
        normalizer_fn=layers.batch_norm,
        normalizer_params=batch_norm_params):

	# Each tuple is (number of channels, kernel size, stride)
        disc = slim.stack(img, slim.conv2d, [(64, [3,3], [2,2]), (128, [3,3], [2,2]),
            (256, [3,3], [2,2]), (512, [3,3], [2,2]), (1024, [3,3], [2,2])], scope="convolution")


    disc = tf.reshape(disc, [batch_size, 4*4*1024])
    disc = slim.fully_connected(disc, 1, activation_fn=None, scope="logits")

    return disc


def main():
    images = YOURDATAHERE # Feed your data here! The program expects batches of 128x128x3 float32 (normalized to be between 0 and 1) images by default
    tf.image_summary("real", images, max_images=1)

    z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z')

    with tf.variable_scope("generator") as scope:
        gen = generator(z)
        tf.image_summary("fake", gen, max_images=1)

    with tf.variable_scope("discriminator") as scope:
        disc_real = discriminator(images)
        scope.reuse_variables()
        disc_fake = discriminator(gen)


    # Define Losses
    disc_real_loss = losses.sigmoid_cross_entropy(disc_real, tf.ones([batch_size, 1]))
    disc_fake_loss = losses.sigmoid_cross_entropy(disc_fake, tf.fill([batch_size, 1], -1.0))

    d_loss = disc_real_loss + disc_fake_loss
    g_loss = losses.sigmoid_cross_entropy(disc_fake, tf.ones([batch_size, 1]))

    tf.scalar_summary("Discriminator_loss_real", disc_real_loss)
    tf.scalar_summary("Discrimintator_loss_fake", disc_fake_loss)
    tf.scalar_summary("Discriminator_loss", d_loss)
    tf.scalar_summary("Generator_loss", g_loss)

    # The paper found RMSProp to work better than Adam or other momentum based methods
    d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
    g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
    g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    # Create training ops
    d_train_op = slim.learning.create_train_op(d_loss, d_optimizer, variables_to_train=d_vars)
    g_train_op = slim.learning.create_train_op(g_loss, g_optimizer, variables_to_train=g_vars)

    # Create clipping ops, thanks to PatrykChrabaszcz for this!
    clip_discriminator = []
    for var in d_vars:
        clip_discriminator.append(tf.assign(var, tf.clip_by_value(var, -c, c)))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(log_dir, sess.graph)
        saver = tf.train.Saver()

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        start = 0
        # If a checkpoint is found, restore what you can. If not, continue
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("Checkpoint found! Restoring...")
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Hackey way to determine what step we're starting on. It feels like there should be some in built function in TensorFlow to do this but I can't find any...
            start = int(ckpt.model_checkpoint_path.split("-")[-1])+1
            print("Restored!")
        else:
            print("No checkpoint found!")

        try:
            current_step = start
            print("Starting training!")
            for itr in xrange(start, max_iterations):

                # As per the reference implementation, the discriminator gets a lot of training early on
                if current_step < 25 or current_step % 500 == 0:
                    diters = 100
                else:
                    diters = d_iters

                # Train discriminator several times
                for i in xrange(diters):
                    # Clip all discriminator weights to be between -c and c
                    if i % clip_per == 0:
                        sess.run(clip_discriminator)
                    batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                    sess.run(d_train_op, feed_dict={z: batch_z})

                # Train generator once
                batch_z = np.random.uniform(-1, 1, [batch_size, z_dim]).astype(np.float32)
                sess.run(g_train_op, feed_dict={z: batch_z})


                # Give the user some feedback
                if itr % sum_per == 0:
                    g_loss_val, d_loss_val, summary_str = sess.run([g_loss, d_loss, summary_op], feed_dict={z: batch_z})
                    print("Step: %d, Generator Loss: %g, Discriminator Loss: %g" % (itr, g_loss_val, d_loss_val))
                    summary_writer.add_summary(summary_str, itr)

                # Every so often save
                if itr % save_per == 0:
                    saver.save(sess, os.path.join(log_dir, "model.ckpt"), global_step=itr)
                current_step = itr

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached!')

        except KeyboardInterrupt:
            print("Ending training...")
            # User terminated with Ctrl-C, save current state
            saver.save(sess, os.path.join(log_dir, "model.ckpt"), global_step=current_step)

        finally:
            coord.request_stop()

        # Done!
        coord.join(threads)


if __name__ == "__main__":
    main()
