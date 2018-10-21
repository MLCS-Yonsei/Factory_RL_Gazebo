import tensorflow as tf
import numpy as np
import copy

class SRL:

    def __init__(self, config):

        # get parameters
        self.epsilon = 0.8
        self.gamma = tf.constant(config.gamma, dtype=tf.float32, name='gamma')

        # get dimensions from configuration
        self.observation_dim = config.observation_dim
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        
        # gpu options
        if config.gpu:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
        else:
            sess_config = None
        self.sess = tf.Session(config = sess_config)

        # placeholder setup
        self.reward = tf.placeholder(tf.float32, [None, 1], name='reward')
        self.done = tf.placeholder(tf.float32, [None, 1], name='done')
        self.action = tf.placeholder(tf.float32, [None]+self.action_dim, name='action')
        self.target_q = tf.placeholder(tf.float32, [None, 1], name='target_q')
        self.target_state = tf.placeholder(
            tf.float32,
            [None]+self.state_dim,
            name='target_state'
        )
        self.var_list = {}

        # observation networks
        self.obs = {}
        for key in config.observation_networks.keys():
            for name in [key, key+'_target']:
                with tf.name_scope(key):
                    self.obs[name] = tf.placeholder(
                        tf.float32,
                        [None]+config.observation_dim[key],
                        name='in'
                    )
                    trainable = False if name.split('_')[-1] == 'target' else True
                    for idx, layer in enumerate(config.observation_networks[key]):
                        self.obs[name] = \
                            _create_layer(
                                self.obs[name],
                                layer,
                                str(idx)+':'+layer['type'],
                                trainable=trainable
                            )

                var_list[name] = \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        self.state = 0.0
        for key in config.observation_networks.keys():
            self.state = tf.add(self.state,self.obs[key])
        self.state = tf.divide(state, len(config.observation_networks), name='state')

        # reinforcement learnign networks
        self.rl = {}
        self.noise_key = {}
        self.noise_dim = {}
        for key in config.rl_networks.keys():
            for name in [key, key+'_target']:
                with tf.name_scope(name):
                    if key =='actor':
                        self.rl[name] = self.state
                    else:
                        self.rl[name] = tf.concat([self.state, self.action], axis=1)
                    trainable = False if name.split('_')[-1] == 'target' else True
                    for idx, layer in enumerate(config.rl_networks[key]):
                        self.rl[name] = \
                            _create_layer(
                                self.rl[name],
                                layer,
                                str(idx)+':'+layer['type'],
                                trainable=trainable
                            )
                            if layer['type'] =='decision':
                                self.noise_key[name] = \
                                    name+'/'+str(idx)+':decision/noise:0'
                                if trainable:
                                    self.noise_dim = layer['shape']

                var_list[name] = \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        self.variables = {var.name:var for var in var_list}
        
        # prediction networks
        self.pred = {}
        for key in config.prediction_networks.keys():
            with tf.name_scope(key):
                self.pred[key] = tf.concat([self.state, self.action], axis=1)
                for idx, layer in enumerate(config.prediction_networks[key]):
                    self.pred[key] = \
                        _create_layer(
                            self.pred[key],
                            layer,
                            str(idx)+':'+layer['type']
                        )

        # state representation loss
        fwd_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.pow(
                    tf.subtract(self.pred['state'], self.target_state),
                    2
                ),
                axis=1
            )
        )

        # reinforcement learning loss
        y = self.reward\
            +tf.multiply(self.gamma, tf.multiply(self.target_q, 1.0-self.done))
        q_loss = \
            tf.reduce_mean(tf.pow(self.rl['critic']-y, 2))\
            +config.l2_penalty*_l2_loss(self.critic_net.var_list)

        # update critic
        self.update_critic = \
            tf.train.AdamOptimizer(learning_rate = config.critic_learning_rate)\
                .minimize(q_loss, var_list = self.critic_net.var_list)

        # update actor
        act_grad_v = tf.gradients(self.rl['critic'], self.action)
        action_gradients = [act_grad_v[0]/tf.to_float(tf.shape(act_grad_v[0])[0])]
        del_Q_a = _gradient_inverter( \
            config.action_bounds, action_gradients, self.rl['actor'])
        parameters_gradients = tf.gradients(
            self.rl['actor'], self.actor_net.var_list, -del_Q_a)
        self.update_actor = tf.train.AdamOptimizer( \
            learning_rate = config.actor_learning_rate) \
            .apply_gradients(zip(parameters_gradients, self.actor_net.var_list))

        # target copy
        self.assign_target = \
            [self.actor_target.variables[var].assign(
                self.actor_net.variables[var.replace('_target', '')]
            ) for var in self.actor_target.variables.keys()]+\
            [self.critic_target.variables[var].assign( \
                self.critic_net.variables[var.replace('_target', '')]
            ) for var in self.critic_target.variables.keys()]
        self.assign_target_soft = \
            [self.actor_target.variables[var].assign(
                config.tau*self.actor_net.variables[var.replace('_target', '')]\
                +(1-config.tau)*self.actor_target.variables[var]
            ) for var in self.actor_target.variables.keys()]+\
            [self.critic_target.variables[var].assign( \
                config.tau*self.critic_net.variables[var.replace('_target', '')]\
                +(1-config.tau)*self.critic_target.variables[var]
            ) for var in self.critic_target.variables.keys()]

        # initialize variables
        self.var_init = tf.global_variables_initializer()
        self.sess.run(self.var_init)
        self.sess.run(self.assign_target)


    def chooseAction(self, observation):

        fd = {}
        for key in self.observation_dim:
            fd[key+'/in:0'] = \
                np.reshape(observation[key], [1]+self.observation_dim[key])
        fd[self.noise_key['actor']] = \
            np.random.normal(
                loc=0.0,
                scale=self.epsilon,
                size=self.noise_dim
            )

        action = self.sess.run(self.rl['actor'], feed_dict=fd)

        return np.reshape(action, self.action_dim)


    def learn(self, batch):

        fd = {}
        for key in self.observation_dim:
            fd[key+'/in:0'] = \
                np.reshape(batch[key+'_1'], [1]+self.observation_dim[key])
        fd[self.noise_key['actor_target']] = np.zeros(self.noise_dim)

        target_action = self.sess.run(self.rl['actor_target'], feed_dict=fd)

        fd[self.action] = target_action

        target_q = self.sess.run(self.rl['critic_target'], feed_dict=fd)

        fd = {}
        for key in self.observation_dim:
            fd[key+'/in:0'] = \
                np.reshape(batch[key+'_0'], [1]+self.observation_dim[key])
        fd[self.noise_key['actor']] = np.zeros(self.noise_dim)
        fd[self.action] = np.reshape(batch['action'], self.action_dim)
        fd[self.target_q] = target_q
        fd[self.target_state]
        fd[self.reward] = np.reshape(batch['reward'], [-1, 1])
        fd[self.done] = np.reshape(batch['done'], [-1, 1])

        self.sess.run([self.update_critic, self.update_actor], feed_dict=fd)

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


def _create_layer(in_, layer, name, trainable=True, eps=None):
    with tf.name_scope(name):
        if layer['type'] == 'dense':
            stddev = 1/np.sqrt(layer['shape'][0])
            with tf.name_scope('dense'):
                out_ = tf.matmul(
                    in_, 
                    tf.Variable(
                        tf.random_normal(layer['shape'], stddev=stddev), 
                        name = 'w', 
                        trainable=trainable
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
                out_ = _activation(layer['activation'], out_)
        elif layer['type'] == 'flatten':
            with tf.name_scope('flatten'):
                out_ = tf.layers.flatten(in_)
                out_ = tf.matmul(
                    out_, 
                    tf.Variable(
                        tf.random_normal([out_.shape[-1]]+layer['shape'][-1], stddev=stddev), 
                        name='weight', 
                        trainable=trainable
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
        elif layer['type'] == 'decision':
            [a_max, a_min] = layer['bounds']
            stddev = 1/np.sqrt(layer['shape'][0])
            with tf.name_scope('decision'):
                out_ = tf.matmul(
                    in_,
                    tf.add(
                        tf.Variable(
                            tf.random_normal(layer['shape'], stddev=stddev), 
                            name = 'w', 
                            trainable=trainable
                        ),
                        tf.placeholder(tf.float32, layer['shape'], name='noise')
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
                out_ = _activation('tanh', out_)
                out_ = tf.multiply(
                    out_,
                    tf.Variable(
                        tf.subtract(a_max, a_min)/2.0,
                        name='a_scale',
                        trainable=False
                    )
                )
                out_ = tf.add(
                    out_,
                    tf.Variable(
                        tf.add(a_max, a_min)/2.0, 
                        name='a_mean', 
                        trainable=False
                    )
                )
                out_ = _activation('tanh', out_)
        else:
            out_ = in_

        return out_


def _activation(f,  in_):
    if f == 'relu':
        return tf.nn.relu(in_)
    if f == 'prelu':
        return tf.nn.leaky_relu(in_)
    if f == 'softplus':
        return tf.nn.softplus(in_)
    if f == 'sigmoid':
        return tf.nn.sigmoid(in_)
    if f == 'tanh':
        return tf.nn.tanh(in_)
    else:
        return in_
        

def _l2_loss(vars):
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
