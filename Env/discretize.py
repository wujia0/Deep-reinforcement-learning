import numpy as np
import tensorflow as tf

def discretize(value,action_dim,n_outputs):
    discretization = tf.round(value)
    discretization = tf.minimum(tf.constant(n_outputs-1, dtype=tf.float32,shape=[1,action_dim]), 
                      tf.maximum(tf.constant(0, dtype=tf.float32,shape=[1,action_dim]), tf.to_float(discretization)))
    return tf.to_int32(discretization)

if __name__=='__main__':
    value=np.array((0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9))
    
    a=discretize(value,value.shape[0],2)
    with tf.Session() as sess:
        print(a.eval())