import os
import keras as k
import tensorflow as tf
from keras import backend as K
from NMT_lib.lang_util import *

class Checkpoint(tf.keras.Model):
#{    
    def __init__(self, checkpoint_dir, optimizer_obj, encoder, decoder):
    #{
        super(Checkpoint, self).__init__()

        self.checkpoint_dir    = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir[:-1], "ckpt")

        self.checkpoint        = tf.train.Checkpoint(optimizer=optimizer_obj, encoder=encoder, decoder=decoder)
    #}
	
    def readCheckpointFile(path):
    #{
        return read_file(path)	
    #}	
	
    def loadCheckpoint(checkpoint_obj, checkpoint_dir, checkpoint_file=None, enable=True):
    #{
        if checkpoint_file:
            data = 'model_checkpoint_path: "' + checkpoint_file + '"\n'\
                   'all_model_checkpoint_paths: "' + checkpoint_file + '"'

            save_file(checkpoint_dir+"checkpoint", data, writeMode= "w")
            checkpoint_obj.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        else:
            # restoring the latest checkpoint in checkpoint_dir
            if enable:
                checkpoint_obj.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        print("Checkpoint loaded.")
    #}
#}