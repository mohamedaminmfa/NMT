import keras as k
import tensorflow as tf
from keras import backend as K

class Optimizer(tf.keras.Model):
#{    
    def __init__(self, optimizer_name, learning_rate, loss_function_api):
    #{
        super(Optimizer, self).__init__()
        self.loss_function_api = loss_function_api
        self.optimizer_name    =  optimizer_name
        self.learning_rate     = learning_rate
		
        self.optimizer         = tf.train.AdamOptimizer()
    #}
        
    def call(self):    
    #{
        learning_rate_obj = tfe.Variable(self.learning_rate, name="learning_rate_obj")
		
        if self.loss_function_api.upper() == "TENSORFLOW-API-LOSS":
        #{
            if self.optimizer_name.upper()   == "ADAM".upper():
                self.optimizer = tf.train.AdamOptimizer(learning_rate_obj)

            elif self.optimizer_name.upper() == "ADAM-MAX".upper():        
                self.optimizer = tf.contrib.opt.AdaMaxOptimizer

            elif self.optimizer_name.upper() == "ADA-GRAD".upper():
                self.optimizer = tf.train.AdagradOptimizer(learning_rate_obj)

            elif self.optimizer_name.upper() == "ADA-DELTA".upper():     
                self.optimizer = tf.train.AdadeltaOptimizer(learning_rate_obj)

            elif self.optimizer_name.upper() == "SGD".upper():
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate_obj)

            elif self.optimizer_name.upper() == "RMS".upper():
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate_obj)

            else:
                self.optimizer = tf.train.AdamOptimizer(learning_rate_obj)

            return self.optimizer
		#}

        elif self.loss_function_api.upper() == "KERAS-API-LOSS":
        #{   
            # When using tensorflow==2.0.0-alpha0

            if self.optimizer_name.upper()   == "ADAM".upper():
                self.optimizer = tf.keras.optimizers.Adam()

            elif self.optimizer_name.upper() == "ADAM-MAX".upper():
                self.optimizer = tf.keras.optimizers.Adamax()

            elif self.optimizer_name.upper() == "ADA-DELTA".upper():
                self.optimizer = tf.keras.optimizers.Adadelta(learning_rate_obj)

            elif self.optimizer_name.upper() == "ADA-GRAD".upper():
                self.optimizer = tf.keras.optimizers.Adagrad(learning_rate_obj)

            elif self.optimizer_name.upper() == "SGD".upper():
                self.optimizer = tf.keras.optimizers.SGD(learning_rate_obj)

            elif self.optimizer_name.upper() == "RMS".upper():
                self.optimizer = tf.keras.optimizers.RMSprop()

            return self.optimizer
        #}		
	#}
#}