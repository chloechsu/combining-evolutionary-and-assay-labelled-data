'''
Converts UniRep model checkpoints to weights
'''

import tensorflow.compat.v1 as tf
import numpy as np
import glob
import os

checkpoint_dir = 'weights/evotuned_release_ckpt/model-13560' 
target_dir = 'weights/evotuned_release' 


def dump_weights(sess, dir_name):
    """
    Saves the weights of the model in dir_name in the format required 
    for loading in this module. Must be called within a tf.Session
    For which the weights are already initialized.
    """
    vs = tf.trainable_variables()
    for v in vs:
        name = v.name
        value = sess.run(v)
        print(name)
        np.save(os.path.join(dir_name,name.replace('/', '_') + ".npy"), np.array(value))


with tf.Session() as sess:
    # Restore variables from disk.
    saver = tf.train.import_meta_graph(checkpoint_dir + '.meta')
    saver.restore(sess, checkpoint_dir)
    print("Variables restored from %s, writing to target dir %s." % (checkpoint_dir, target_dir))
    print("Saved variables:")
    dump_weights(sess, dir_name=target_dir)
