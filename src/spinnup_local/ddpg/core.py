import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def mlp_bn(x,phase, hidden_sizes=(32,), activation=tf.nn.tanh, output_activation=None):
    x = tf.contrib.layers.batch_norm(x,is_training = phase,scope = None)
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h)
        x = activation(x)
        x = tf.contrib.layers.batch_norm(x, is_training=phase, scope=None)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def critic(x,a):
    x = tf.layers.dense(x, units=400, activation=tf.nn.relu)
    x = tf.concat([x, a], axis=-1) # this assumes observation and action can be concatenated
    x = tf.layers.dense(x, units=300, activation=tf.nn.relu)
    x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
    return x


"""
Actor-Critics
"""
def mlp_actor_critic(x, a, hidden_sizes=(400,300), activation=tf.nn.relu,
                     output_activation=tf.tanh, action_space=None,batch_normalization = False,phase = None):
    act_dim = a.shape.as_list()[-1]
    #act_limit = action_space.high[0]
    with tf.variable_scope('pi'):
        #pi = act_limit * mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
        if(batch_normalization):
            pi = mlp_bn(x,phase, list(hidden_sizes)+[act_dim], activation, output_activation)
        else:
            pi = mlp(x, list(hidden_sizes)+[act_dim], activation, output_activation)
    with tf.variable_scope('q'):
        if(batch_normalization):
            q = tf.squeeze(mlp_bn(tf.concat([x,a], axis=-1),phase, list(hidden_sizes)+[1], activation, None), axis=1)
        else:
            q = tf.squeeze(mlp(tf.concat([x,a], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        #q = tf.squeeze(critic(x,a), axis=1)
    with tf.variable_scope('q', reuse=True):
        if(batch_normalization):
            q_pi = tf.squeeze(mlp_bn(tf.concat([x,pi], axis=-1),phase, list(hidden_sizes)+[1], activation, None), axis=1)
        else:
            q_pi = tf.squeeze(mlp(tf.concat([x,pi], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
        #q_pi = tf.squeeze(critic(x,pi), axis=1)
    return pi, q, q_pi
