import os
import keras as k
import tensorflow as tf

from SL.lib.NMT_lib.lang_util import *

class Checkpoint(tf.keras.Model):
#{    
    def __init__(self, checkpoint_dir, optimizer_obj, encoder, decoder):
    #{
        super(Checkpoint, self).__init__()

        self.checkpoint_dir    = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

        self.checkpoint        = tf.train.Checkpoint(optimizer=optimizer_obj, encoder=encoder, decoder=decoder)
    #}
	
    def readCheckpointFile(self, path):
    #{
        return read_file(path)	
    #}	
	
    def loadCheckpoint(self, checkpoint_dir, checkpoint_file=None, enable=True):
    #{
        if checkpoint_file != None:
            #data = 'model_checkpoint_path: "' + checkpoint_file + '"\nall_model_checkpoint_paths: "' + checkpoint_file + '"'

            #save_file(checkpoint_dir+"checkpoint", data, writeMode= "w")
            self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir, checkpoint_file))

        else:
            # restoring the latest checkpoint in checkpoint_dir
            if enable:
                self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        print("Checkpoint loaded.")
    #}
#}