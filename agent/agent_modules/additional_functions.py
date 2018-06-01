import tensorflow as tf

def l2_regularizer(vars):
    loss=0
    for var in vars:
        loss+=tf.reduce_sum(tf.pow(var,2))
    return loss/2.0

def gradient_inverter(action_bounds,action_gradients,actions):
    action_dim=len(action_bounds[0])
    pmax=tf.constant(action_bounds[0],dtype=tf.float32)
    pmin=tf.constant(action_bounds[1],dtype=tf.float32)
    prange=tf.constant([x-y for x,y in zip(action_bounds[0],action_bounds[1])],dtype=tf.float32)
    pdiff_max=tf.div(-actions+pmax,prange)
    pdiff_min=tf.div(actions-pmin,prange)
    zeros_act_grad_filter=tf.zeros([action_dim])
    act_grad=tf.placeholder(tf.float32,[None,action_dim])
    # grad_inverter = tf.select(tf.greater(act_grad, zeros_act_grad_filter), tf.mul(act_grad, pdiff_max), tf.mul(act_grad, pdiff_min))        
    return tf.where(tf.greater(act_grad,zeros_act_grad_filter),tf.multiply(act_grad,pdiff_max),tf.multiply(act_grad,pdiff_min))   