import tensorflow as tf
import numpy as np
import copy

class SRL:

    def __init__(self, config):

        self.epsilon = 0.8
        self.action_dim = config.action_dim
        self.action_scale = np.reshape(config.action_bounds[0], [1, config.action_dim])
        self.gamma = tf.constant(config.gamma, dtype=tf.float32, name='gamma')
        
        if config.gpu:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
        else:
            sess_config = None
        self.sess = tf.Session(config = sess_config)

        self.var_init = tf.global_variables_initializer()

        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
        self.done = tf.placeholder(tf.float32, [None, 1], name='done')
        self.target_q = tf.placeholder(tf.float32, [None, 1], name='target_q')
        self.target_state = tf.placeholder(
            tf.float32,
            [None]+config.state_dim,
            name='target_state'
        )

        # observation networks
        self.obs = {}
        for key in config.observation_networks.keys():
            with tf.name_scope(key):
                self.obs[key] = tf.placeholder(
                    tf.float32,
                    [None]+config.observation_dim[key],
                    name='o'
                )
                for idx, layer in enumerate(config.observation_networks[key]):
                    self.obs[key] = _create_layer(self.obs, layer, layer['type']+str(idx))

        # prediction networks
        for key in config.prediction_networks.keys():
            pass

        # reinforcement learnign networks
        for key in config.rl_networks.keys():
            pass

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
        self.variables = {var.name:var for var in self.var_list}
        # self.noise = tf.placeholder(tf.float32, [None, config.action_dim])
        # build network
        self.actor_net = Build_AC(self.sess, config, 'actor_net')
        self.actor_target = Build_AC(self.sess, config, 'actor_target')
        self.critic_net = Build_AC(self.sess, config, 'critic_net')
        self.critic_target = Build_AC(self.sess, config, 'critic_target')
        # update critic
        y = self.reward+tf.multiply(self.gamma, tf.multiply(self.target_q, 1.0-self.done))
        # y = self.reward+tf.multiply(self.gamma, self.target_q)
        q_loss = tf.reduce_sum(tf.pow(self.critic_net.out_-y, 2))/config.batch_size+ \
            config.l2_penalty*_l2_regularizer(self.critic_net.var_list)
        self.update_critic = tf.train.AdamOptimizer( \
            learning_rate = config.critic_learning_rate).minimize(q_loss, var_list = self.critic_net.var_list)
        # update actor
        act_grad_v = tf.gradients(self.critic_net.out_, self.critic_net.action)
        action_gradients = [act_grad_v[0]/tf.to_float(tf.shape(act_grad_v[0])[0])]
        del_Q_a = _gradient_inverter( \
            config.action_bounds, action_gradients, self.actor_net.out_)
        parameters_gradients = tf.gradients(
            self.actor_net.out_, self.actor_net.var_list, -del_Q_a)
        self.update_actor = tf.train.AdamOptimizer( \
            learning_rate = config.actor_learning_rate) \
            .apply_gradients(zip(parameters_gradients, self.actor_net.var_list))
        # target copy
        self.assign_target =  \
            [self.actor_target.variables[var].assign( \
                self.actor_net.variables[var.replace('_target', '_net')] \
            ) for var in self.actor_target.variables.keys()]+ \
            [self.critic_target.variables[var].assign( \
                self.critic_net.variables[var.replace('_target', '_net')] \
            ) for var in self.critic_target.variables.keys()]
        self.assign_target_soft =  \
            [self.actor_target.variables[var].assign( \
                config.tau*self.actor_net.variables[var.replace('_target', '_net')]+ \
                (1-config.tau)*self.actor_target.variables[var] \
            ) for var in self.actor_target.variables.keys()]+ \
            [self.critic_target.variables[var].assign( \
                config.tau*self.critic_net.variables[var.replace('_target', '_net')]+ \
                (1-config.tau)*self.critic_target.variables[var] \
            ) for var in self.critic_target.variables.keys()]
        # initialize variables
        self.var_init = tf.global_variables_initializer()
        self.sess.run(self.var_init)
        self.sess.run(self.assign_target)
        self.a_scale, self.a_mean = self.sess.run(
            [self.actor_net.a_scale, self.actor_net.a_mean])

    def chooseAction(self, state):
        state_vector = np.reshape(state['vector'], [1, -1])
        state_rgbd = np.reshape(state['rgbd'], [1, 96, 128, 7])
        action = self.actor_net.evaluate(state_vector, state_rgbd)
        # action = self.sess.run(self.actor_net.out_before_activation,  \
        #     feed_dict = {self.actor_net.state_vector:state_vector,  \
        #                self.actor_net.state_rgbd:state_rgbd})
        # action = self.a_scale* \
        #        np.tanh(action+self.epsilon*np.random.randn(1, self.action_dim))+ \
        #        self.a_mean
        action = action+self.epsilon*self.action_scale*np.random.randn(1, self.action_dim)
        return np.reshape(action, [self.action_dim])

    def learn(self, batch):
        vector0 = np.reshape(batch['vector0'], self.vector_dim)
        rgbd0 = np.reshape(batch['rgbd0'], self.rgbd_dim)
        vector1 = np.reshape(batch['vector1'], self.vector_dim)
        rgbd1 = np.reshape(batch['rgbd1'], self.rgbd_dim)
        action0 = np.reshape(batch['action0'], [-1, self.action_dim])
        reward = np.reshape(batch['reward'], [-1, 1])
        done = np.reshape(batch['done'], [-1, 1])
        target_action = self.actor_target.evaluate(vector1, rgbd1)
        target_q = self.critic_target.evaluate(vector1, rgbd1, action = target_action)
        self.sess.run(self.update_critic,  \
                      feed_dict = {self.critic_net.state_vector:vector0,  \
                                 self.critic_net.state_rgbd:rgbd0,  \
                                 self.critic_net.action:action0,  \
                                 self.reward:reward,  \
                                 self.target_q:target_q,  \
                                 self.done:done})
        self.sess.run(self.update_actor,  \
                      feed_dict = {self.critic_net.state_vector:vector0,  \
                                 self.critic_net.state_rgbd:rgbd0,  \
                                 self.critic_net.action:action0,  \
                                 self.actor_net.state_vector:vector0,  \
                                 self.actor_net.state_rgbd:rgbd0})
        self.sess.run(self.assign_target_soft)

    def reset(self):
        self.sess.run(self.var_init)
    
    def load(self, saved_variables):
        self.sess.run( \
            [self.actor_net.variables[var].assign(saved_variables[var]) \
                for var in self.actor_net.variables.keys()]+ \
            [self.critic_net.variables[var].assign(saved_variables[var]) \
                for var in self.critic_net.variables.keys()]+ \
            self.assign_target)
    
    def return_variables(self):
        return dict({name:self.sess.run(name) \
                    for name in self.actor_net.variables.keys()},  \
               **{name:self.sess.run(name) \
                    for name in self.critic_net.variables.keys()})


class Build_AC(object):

    def __init__(self, sess, layers, name, *args):
        self.name = name
        self.sess = sess
        layers = copy.copy(config.layers)
        self.trainable = False if name.split('_')[1] == 'target' else True
        with tf.name_scope(name):
            self.state_vector = tf.placeholder(tf.float32, config.vector_dim)
            self.state_rgbd = tf.placeholder(tf.float32, config.rgbd_dim)
            layers['merge'][0][0] =  \
                    layers['rgbd'][-1][-1]* \
                    config.rgbd_dim[1]* \
                    config.rgbd_dim[2]/ \
                    2**(2*len(layers['rgbd']))+ \
                    layers['vector'][-1][-1]
            if name[0] == 'a':
                # config.action_dim = len(config.action_bounds[0])
                self.a_scale = tf.subtract(
                    config.action_bounds[0], config.action_bounds[1])/2.0
                self.a_mean = tf.add(
                    config.action_bounds[0], config.action_bounds[1])/2.0
            else:
                layers['merge'][0][0] += config.action_dim
                self.action = tf.placeholder(tf.float32, [None, config.action_dim])
            for item in layers.keys():
                for idx, shape in enumerate(layers[item]):
                    self.create_variable(shape, item+str(idx))
            if name[0] == 'c':
                self.create_variable([layers['merge'][-1][-1], 1], 'output')
            else:
                self.create_variable([layers['merge'][-1][-1], config.action_dim], 'output')
            self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
            self.variables = {var.name:var for var in self.var_list}
            out_vector = self.state_vector
            for layer in range(len(layers['vector'])):
                out_vector = self.fc(out_vector, 'vector'+str(layer))
            out_rgbd = self.state_rgbd
            for layer in range(len(layers['rgbd'])):
                out_rgbd = self.conv(out_rgbd, 'rgbd'+str(layer))
            out_rgbd = tf.reshape(out_rgbd,  \
                [
                    -1, 
                    layers['rgbd'][-1][-1]* \
                    config.rgbd_dim[1]* \
                    config.rgbd_dim[2]/ \
                    2**(2*len(layers['rgbd']))
                ])
            out_ = tf.concat([out_vector, out_rgbd], 1)
            if name[0] == 'c':
                out_ = tf.concat([out_, self.action], 1)
            for layer in range(len(layers['merge'])):
                out_ = self.fc(out_, 'merge'+str(layer))
            out_ = tf.matmul(out_, self.variables[self.name+'/output/w:0'])
            if name.split('_')[0] == 'actor':
                self.out_before_activation = out_
                self.out_ = tf.multiply(tf.tanh(out_), self.a_scale)+self.a_mean
            else:
                self.out_ = out_

    def evaluate(self, feed_dict):
        return self.sess.run(self.out_, feed_dict = feed_dict)    


def _create_layer(in_, layer, name, trainable=True):
    with tf.name_scope(name):
        if layer['type'] == 'dense':
            stddev = 1/np.sqrt(layer['shape'][0])
            with tf.name_scope('dense'):
                out_ = tf.matmul(
                    in_, 
                    tf.Variable(
                        tf.random_normal(layer['shape'], stddev=stddev), 
                        name = 'w', 
                        trainable = trainable
                    )
                )
                out_ = tf.add(
                    out_,
                    tf.Variable(
                        tf.random_normal(layer['shape'][-1], stddev=stddev), 
                        name='b', 
                        trainable=trainable
                    )
                )
                out_ = _activation(layer[1], out_)
        elif layer['type'] == 'flatten':
            with tf.name_scope('flatten'):
                out_ = tf.layers.flatten(in_)
                out_ = tf.matmul(
                    out_, 
                    tf.Variable(
                        tf.random_normal([out_.shape[-1]]+layer['shape'][-1], stddev=stddev), 
                        name='weight', 
                        trainable = trainable
                    )
                )
                out_ = tf.add(
                    out_,
                    tf.Variable(
                        tf.random_normal(layer['shape'][-1], stddev=stddev), 
                        name='bias', 
                        trainable=trainable
                    )
                )
                out_ = _activation(layer['activation'], out_)
        elif layer['type'] == 'conv2d':
            stddev = 1/np.sqrt(layer['shape'][0]*layer['shape'][1]*layer['shape'][2])
            with tf.name_scope('conv2d'):
                out_ = tf.nn.conv2d(
                    in_, 
                    tf.Variable(
                        tf.random_normal(shape, stddev=stddev),
                        name='filter',
                        trainable=trainable
                    ),
                    strides=[1, 1, 1, 1], 
                    padding='SAME'
                )
                out_ = tf.add(
                    out_,
                    tf.Variable(
                        tf.random_normal(shape[3], stddev=stddev),
                        name='bias',
                        trainable=trainable)
                )
                out_ = _activation(layer['activation'], out_)
                if layer['pool']:
                    out_ = tf.nn.max_pool(out_, layer['pool'], layer['pool'],padding='SAME')
        else:
            out_ = in_

        return out_


def _activation(type,  in_):
    if type == 'relu':
        return tf.nn.relu(in_)
    if type == 'prelu':
        return tf.nn.leaky_relu(in_)
    if type == 'softplus':
        return tf.nn.softplus(in_)
    if type == 'sigmoid':
        return tf.nn.sigmoid(in_)
    if type == 'tanh':
        return tf.nn.tanh(in_)
    else:
        return in_
        

def _l2_regularizer(vars):
    loss = 0
    for var in vars:
        loss += tf.reduce_sum(tf.pow(var, 2))
    return loss/2.0

def _gradient_inverter(action_bounds, action_gradients, actions):
    action_dim = len(action_bounds[0])
    pmax = tf.constant(action_bounds[0], dtype=tf.float32)
    pmin = tf.constant(action_bounds[1], dtype=tf.float32)
    prange = tf.constant(
        [x-y for x, y in zip(action_bounds[0],
        action_bounds[1])],
        dtype = tf.float32
    )
    pdiff_max = tf.div(-actions+pmax, prange)
    pdiff_min = tf.div(actions-pmin, prange)
    zeros_act_grad_filter = tf.zeros([action_dim])       
    return tf.where(
        tf.greater(action_gradients, zeros_act_grad_filter),
        tf.multiply(action_gradients, pdiff_max),
        tf.multiply(action_gradients, pdiff_min)
    )
