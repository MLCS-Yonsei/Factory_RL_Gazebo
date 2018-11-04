import tensorflow as tf
import numpy as np
import copy

class DDPG:
    def __init__(self, config):
        if config.gpu:
            sess_config=tf.ConfigProto()
            sess_config.gpu_options.allow_growth=True
        else:
            sess_config=None
        self.epsilon=0.5
        self.action_dim=config.action_dim
        self.state_dim = config.state_dim
        self.gamma=tf.constant(config.gamma,dtype=tf.float32,name='gamma')
        self.sess=tf.Session(config=sess_config)
        self.var_init=tf.global_variables_initializer()
        self.state=tf.placeholder(tf.float32,[None]+self.state_dim)
        self.state_target=tf.placeholder(tf.float32,[None]+self.state_dim)
        self.action=tf.placeholder(tf.float32,[None,3])
        self.reward=tf.placeholder(tf.float32,[None,1])
        self.done=tf.placeholder(tf.float32,[None,1])
        self.target_q=tf.placeholder(tf.float32,[None,1])
        self.a_mean = np.add(config.action_max,config.action_min)/2.0
        self.a_scale = np.subtract(config.action_max,config.action_min)/2.0
        self.var_list = {}
        
        # reinforcement learnign networks
        name = 'actor'
        with tf.name_scope(name):
            actor = self.state
            for idx, layer in enumerate(config.networks['actor']):
                actor = \
                    _create_layer(
                        actor,
                        layer,
                        str(idx),
                    )
            with tf.name_scope('decision'):
                self.noise_dim = config.networks['actor'][-1]['shape'][-1:]+self.action_dim
                self.noise_stddev = 1/np.sqrt(self.noise_dim[0])
                w = tf.Variable(
                    tf.random_normal(self.noise_dim, stddev=self.noise_stddev),
                    name='weight',
                )
                b = tf.Variable(
                    tf.random_normal([1]+self.noise_dim[-1:], stddev=self.noise_stddev), 
                    name='bias', 
                )
                self.action_noise = tf.placeholder(tf.float32, self.noise_dim, name='noise')
                self.actor = tf.add(tf.matmul(actor, w), b)
                self.actor = \
                    tf.add(
                        tf.multiply(tf.nn.tanh(self.actor), self.a_scale),
                        self.a_mean
                    )
                self.noisy_action = \
                    tf.add(tf.matmul(actor, tf.add(w, self.action_noise)), b)
                self.noisy_action = \
                    tf.add(
                        tf.multiply(tf.nn.tanh(self.noisy_action), self.a_scale),
                        self.a_mean
                    )

        self.var_list[name] = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        
        name = 'actor_target'
        with tf.name_scope(name):
            actor_target = self.state_target
            for idx, layer in enumerate(config.networks['actor']):
                actor_target = \
                    _create_layer(
                        actor_target,
                        layer,
                        str(idx),
                    )
            with tf.name_scope('decision'):
                w = tf.Variable(
                    tf.random_normal(self.noise_dim, stddev=self.noise_stddev),
                    name='weight',
                    trainable=False
                )
                b = tf.Variable(
                    tf.random_normal([1]+self.noise_dim[-1:], stddev=self.noise_stddev), 
                    name='bias', 
                    trainable=False
                )
                self.actor_target = tf.add(tf.matmul(actor_target, w), b)
                self.actor_target = \
                    tf.add(
                        tf.multiply(tf.nn.tanh(self.actor_target), self.a_scale),
                        self.a_mean
                    )
        self.var_list[name] = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        
        name = 'critic'
        with tf.name_scope(name):
            self.critic = tf.concat([self.state, self.action], axis=1)
            for idx, layer in enumerate(config.networks['critic']):
                self.critic = \
                    _create_layer(
                        self.critic,
                        layer,
                        str(idx),
                    )

        self.var_list[name] = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)

        name = 'critic_target'
        with tf.name_scope(name):
            self.critic_target = tf.concat([self.state_target, self.action], axis=1)
            for idx, layer in enumerate(config.networks['critic']):
                self.critic_target = \
                    _create_layer(
                        self.critic_target,
                        layer,
                        str(idx),
                        trainable=False
                    )

        self.var_list[name] = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        self.vars_for_copy = {
            var.name:var for var in \
                self.var_list['actor']+ \
                self.var_list['critic']+ \
                self.var_list['actor_target']+ \
                self.var_list['critic_target']
        }
        self.target_keys = [
            key for key in self.vars_for_copy.keys() if len(key.split('target'))>1
        ]
        self.var_list['graph'] = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name:var for var in self.var_list['graph']}

        # update critic
        y=self.reward+tf.multiply(self.gamma,tf.multiply(self.target_q,1.0-self.done))
        # y=self.reward+tf.multiply(self.gamma,self.target_q)
        q_loss=tf.reduce_sum(tf.pow(self.critic-y,2))/config.batch_size+ \
            config.l2_penalty*_l2_loss(self.var_list['critic'])
        self.update_critic=tf.train.AdamOptimizer( \
            learning_rate=config.critic_learning_rate).minimize(q_loss,var_list=self.var_list['critic'])
        # update actor
        act_grad_v=tf.gradients(self.critic,self.action)
        action_gradients=[act_grad_v[0]/tf.to_float(tf.shape(act_grad_v[0])[0])]
        del_Q_a=_gradient_inverter( \
            [config.action_max, config.action_min],action_gradients,self.actor)
        parameters_gradients=tf.gradients(
            self.actor,self.var_list['actor'],-del_Q_a)
        self.update_actor=tf.train.AdamOptimizer( \
            learning_rate=config.actor_learning_rate) \
            .apply_gradients(zip(parameters_gradients,self.var_list['actor']))
        # target copy
        self.assign_target = [
            self.vars_for_copy[var].assign(
                self.vars_for_copy[var.replace('_target', '')]
            ) for var in self.target_keys
        ]
        self.assign_target_soft = [
            self.vars_for_copy[var].assign(
                config.tau*self.vars_for_copy[var.replace('_target', '')]\
                +(1-config.tau)*self.vars_for_copy[var]
            ) for var in self.target_keys
        ]
        # initialize variables
        self.var_init=tf.global_variables_initializer()
        self.sess.run(self.var_init)
        self.sess.run(self.assign_target)

    def chooseAction(self,state):
        state=np.reshape(state,[1,-1])
        action=self.sess.run(
            self.noisy_action,
            feed_dict={
                self.state:state,
                self.action_noise:np.random.normal(
                    loc=0.0,
                    scale=self.epsilon*self.noise_stddev,
                    size=self.noise_dim
                )
            }
        )
        return np.reshape(action,self.action_dim)

    def learn(self,batch):
        state0=np.reshape(batch['state_0'],[-1]+self.state_dim)
        state1=np.reshape(batch['state_1'],[-1]+self.state_dim)
        action=np.reshape(batch['action'],[-1]+self.action_dim)
        reward=np.reshape(batch['reward'],[-1,1])
        done=np.reshape(batch['done'],[-1,1])
        target_action=self.sess.run(self.actor_target,feed_dict={self.state_target:state1})
        target_q=self.sess.run(
            self.critic_target,
            feed_dict={
                self.state_target:state1,
                self.action:target_action
            }
        )
        self.sess.run(self.update_critic, \
                      feed_dict={self.state:state0, \
                                 self.action:action, \
                                 self.reward:reward, \
                                 self.target_q:target_q, \
                                 self.done:done})
        self.sess.run(self.update_actor, \
                      feed_dict={self.state:state0, \
                                 self.action:action,})
        self.sess.run(self.assign_target_soft)

    def reset(self):
        self.sess.run(self.var_init)
    
    def load(self,saved_variables):
        self.sess.run(
            [self.vars[var].assign(saved_variables[var]) for var in self.vars.keys()]
        )
        self.sess.run(self.assign_target)
    
    def return_variables(self):

        return {name:self.sess.run(name) for name in self.vars}



def _create_layer(in_, layer, name, trainable=True, eps=None):
    with tf.name_scope(name):
        if layer['type'] == 'dense':
            stddev = 1/np.sqrt(layer['shape'][0])
            with tf.name_scope('dense'):
                out_ = tf.matmul(
                    in_, 
                    tf.Variable(
                        tf.random_normal(layer['shape'], stddev=stddev), 
                        name = 'weight', 
                        trainable=trainable
                    )
                )
                out_ = tf.add(
                    out_,
                    tf.Variable(
                        tf.random_normal([1]+layer['shape'][-1:], stddev=stddev), 
                        name='bias', 
                        trainable=trainable
                    )
                )
                out_ = _activation(layer['activation'], out_)
        elif layer['type'] == 'flatten':
            with tf.name_scope('flatten'):
                out_ = tf.layers.flatten(in_)
                stddev = 1/np.sqrt(out_.shape[-1].value)
                out_ = tf.matmul(
                    out_, 
                    tf.Variable(
                        tf.random_normal([out_.shape[-1].value]+layer['shape'][-1:], stddev=stddev), 
                        name='weight', 
                        trainable=trainable
                    )
                )
                out_ = tf.add(
                    out_,
                    tf.Variable(
                        tf.random_normal(layer['shape'][-1:], stddev=stddev), 
                        name='bias', 
                        trainable=trainable
                    )
                )
                out_ = _activation(layer['activation'], out_)
        elif layer['type'] == 'conv':
            stddev = 1/np.sqrt(layer['shape'][0]*layer['shape'][1]*layer['shape'][2])
            with tf.name_scope('conv'):
                out_ = tf.nn.conv2d(
                    in_, 
                    tf.Variable(
                        tf.random_normal(layer['shape'], stddev=stddev),
                        name='filter',
                        trainable=trainable
                    ),
                    strides=layer['strides'], 
                    padding='SAME'
                )
                out_ = tf.add(
                    out_,
                    tf.Variable(
                        tf.random_normal(layer['shape'][-1:], stddev=stddev),
                        name='bias',
                        trainable=trainable
                    )
                )
                out_ = _activation(layer['activation'], out_)
                if layer['pool'] !='None':
                    out_ = tf.nn.max_pool(out_, layer['pool'], layer['pool'], padding='SAME')
        else:
            out_ = in_

        return out_


def _activation(f, in_):
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
