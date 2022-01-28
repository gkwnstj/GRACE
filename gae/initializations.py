import tensorflow as tf
import numpy as np

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """

##    tf.random.set_seed(6)      ##############
    
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    #initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
    #                            maxval=init_range, dtype=tf.float32)
    initial = tf.random.uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)    # seed=6

##    print(input_dim)
##    print(output_dim)
##    print(initial)
    
    return tf.Variable(initial, name=name)

