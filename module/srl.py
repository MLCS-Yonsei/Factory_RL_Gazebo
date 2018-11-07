import tensorflow as tf
import numpy as np
import copy

class SRL:

    def __init__(self, config):

        # get parameters
        self.epsilon = 0.1
        self.gamma = tf.constant(config.gamma, dtype=tf.float32, name='gamma')
        self.batch_size = config.batch_size

        # get dimensions from configuration
        self.observation_dim = config.observation_dim
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.a_mean = np.add(config.action_max,config.action_min)/2.0
        self.a_scale = np.subtract(config.action_max,config.action_min)/2.0
        
        # gpu options
        if config.gpu:
            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
        else:
            sess_config = None
        
        # Tensorflow session
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
        self.obs_in = {}
        group = 'obs'
        with tf.name_scope(group):
            for key in config.observation_networks.keys():
                for name in [key, key+'_target']:
                    with tf.name_scope(name):
                        trainable = False if name.split('_')[-1] == 'target' else True
                        self.obs_in[name] = tf.placeholder(
                            tf.float32,
                            [None]+config.observation_dim[key],
                            name='in'
                        )
                        self.obs[name] = self.obs_in[name]
                        for idx, layer in enumerate(config.observation_networks[key]):
                            self.obs[name] = \
                                _create_layer(
                                    self.obs[name],
                                    layer,
                                    str(idx),
                                    trainable=trainable
                                )

                    self.var_list[name] = \
                        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group+'/'+name)
            self.var_list[group] = \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group)

        # merge features to state
        for num, key in enumerate(config.observation_networks.keys()):
            if num == 0:
                self.state = self.obs[key]
                self.state_target = self.obs[key+'_target']
            else:
                self.state = tf.add(self.state, self.obs[key])
                self.state_target = tf.add(self.state_target, self.obs[key+'_target'])
        self.state = tf.divide(
            self.state,
            tf.constant(len(config.observation_networks.keys()), dtype=tf.float32),
            name='state'
        )
        self.state_target = tf.divide(
            self.state_target,
            tf.constant(len(config.observation_networks.keys()), dtype=tf.float32),
            name='state_target'
        )

        # reinforcement learnign networks
        group = 'rl'
        with tf.name_scope(group):
            name = 'actor'
            with tf.name_scope(name):
                actor = self.state
                for idx, layer in enumerate(config.rl_networks['actor']):
                    actor = \
                        _create_layer(
                            actor,
                            layer,
                            str(idx),
                        )
                with tf.name_scope('decision'):
                    self.noise_dim = config.rl_networks['actor'][-1]['shape'][-1:]+self.action_dim
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
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group+'/'+name)
            
            name = 'actor_target'
            with tf.name_scope(name):
                actor_target = self.state_target
                for idx, layer in enumerate(config.rl_networks['actor']):
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
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group+'/'+name)
            
            name = 'critic'
            with tf.name_scope(name):
                self.critic = tf.concat([self.state, self.action], axis=1)
                for idx, layer in enumerate(config.rl_networks['critic']):
                    self.critic = \
                        _create_layer(
                            self.critic,
                            layer,
                            str(idx),
                        )

            self.var_list[name] = \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group+'/'+name)

            name = 'critic_target'
            with tf.name_scope(name):
                self.critic_target = tf.concat([self.state_target, self.action], axis=1)
                for idx, layer in enumerate(config.rl_networks['critic']):
                    self.critic_target = \
                        _create_layer(
                            self.critic_target,
                            layer,
                            str(idx),
                            trainable=False
                        )

            self.var_list[name] = \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group+'/'+name)

            self.var_list[group] = \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group)
        self.vars_for_copy = {var.name:var for var in self.var_list['rl']+self.var_list['obs']}
        self.target_keys = [key for key in self.vars_for_copy.keys() if len(key.split('target'))>1]
        
        # prediction networks
        self.pred = {}
        group = 'pred'
        with tf.name_scope(group):
            for key in config.prediction_networks.keys():
                with tf.name_scope(key):
                    self.pred[key] = tf.concat([self.state, self.action], axis=1)
                    for idx, layer in enumerate(config.prediction_networks[key]):
                        self.pred[key] = \
                            _create_layer(
                                self.pred[key],
                                layer,
                                str(idx)
                            )

                self.var_list[key] = \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group+'/'+key)
            self.var_list[group] = \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=group)

        self.var_list['graph'] = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.vars = {var.name:var for var in self.var_list['graph']}

        # state representation loss
        rew_loss = tf.reduce_mean(
            tf.pow(
                tf.subtract(self.pred['reward'], self.reward),
                2
            )
        )
        tf.summary.scalar('reward_prediction_loss', rew_loss)
        fwd_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.pow(
                    tf.subtract(self.pred['state'], self.target_state),
                    2
                ),
                axis=1
            )
        )
        tf.summary.scalar('state_prediction_loss', fwd_loss)
        inv_loss = None # impossible to solve on continuous domain
        slow_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.pow(
                    tf.subtract(self.state, self.target_state),
                    2
                ),
                axis=1
            )
        )
        tf.summary.scalar('slowness_loss', slow_loss)
        # div_loss = tf.reduce_mean(
        #     tf.exp(
        #         -tf.reduce_sum(
        #             tf.pow(
        #                 tf.subtract(
        #                     self.state,
        #                     tf.reduce_mean(self.state, axis=0)
        #                 ),
        #                 2
        #             ),
        #             axis=1
        #         )
        #     )
        # )
        # tf.summary.scalar('diversity_loss', div_loss)
        srl_loss = config.c_srl*(
            config.c_rew*rew_loss+\
            config.c_slow*slow_loss+\
            # config.c_div*div_loss+\
            config.c_fwd*fwd_loss
        )

        # reinforcement learning loss
        y = self.reward \
            +tf.multiply(self.gamma, tf.multiply(self.target_q, 1.0-self.done))
        q_loss = \
            tf.reduce_mean(tf.pow(self.critic-y, 2))\
            +config.l2_penalty*_l2_loss(self.var_list['critic'])
        tf.summary.scalar('Q_loss', q_loss)

        self.merged = tf.summary.merge_all()

        # update all
        self.update_all = \
            tf.train.AdamOptimizer(learning_rate = config.critic_learning_rate)\
                .minimize(
                    q_loss+srl_loss,
                    var_list = \
                        self.var_list['obs']+\
                        self.var_list['critic']+\
                        self.var_list['pred']
                )

        # update critic
        self.update_critic = \
            tf.train.AdamOptimizer(learning_rate = config.critic_learning_rate)\
                .minimize(
                    q_loss,
                    var_list = \
                        self.var_list['obs']+\
                        self.var_list['critic']
                )

        # update srl
        self.update_srl = \
            tf.train.AdamOptimizer(learning_rate = config.srl_learning_rate)\
                .minimize(
                    srl_loss,
                    var_list = \
                        self.var_list['obs']+\
                        self.var_list['pred']
                )

        # update actor
        act_grad_v = tf.gradients(self.critic, self.action)
        action_gradients = [act_grad_v[0]/tf.to_float(tf.shape(act_grad_v[0])[0])]
        del_Q_a = _gradient_inverter(
            [config.action_max, config.action_min],
            action_gradients,
            self.actor
        )
        parameters_gradients = tf.gradients(
            self.actor, self.var_list['actor'], -del_Q_a)
        self.update_actor = tf.train.AdamOptimizer( \
            learning_rate = config.actor_learning_rate) \
            .apply_gradients(zip(parameters_gradients, self.var_list['actor']))

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
        self.var_init = tf.global_variables_initializer()
        self.sess.run(self.var_init)
        self.sess.run(self.assign_target)

        # Tensorboard setup
        self.epoch = 0
        self.writer = tf.summary.FileWriter(config.summary_dir, self.sess.graph)


    def chooseAction(self, observation):

        fd = {}
        for key in self.observation_dim.keys():
            fd[self.obs_in[key]] = \
                self.normalize_obs(observation[key], key, 1)

        fd[self.action_noise] = \
            np.random.normal(
                loc=0.0,
                scale=self.epsilon*self.noise_stddev,
                size=self.noise_dim
            )

        action = self.sess.run(self.noisy_action, feed_dict=fd)

        return np.reshape(action, self.action_dim)


    def learn(self, batch):

        self.epoch += 1

        fd = {}
        for key in self.observation_dim.keys():
            fd[self.obs_in[key+'_target']] = \
                self.normalize_obs(batch[key+'_1'], key, self.batch_size)

        target_action, target_state = \
            self.sess.run([self.actor_target, self.state_target], feed_dict=fd)

        fd[self.action] = target_action

        target_q = self.sess.run(self.critic_target, feed_dict=fd)

        fd.clear()
        for key in self.observation_dim:
            fd[self.obs_in[key]] = \
                self.normalize_obs(batch[key+'_0'], key, self.batch_size)
        fd[self.action] = np.reshape(
            batch['action'], [self.batch_size]+self.action_dim)
        fd[self.target_q] = target_q
        fd[self.reward] = np.reshape(batch['reward'], [self.batch_size, 1])
        fd[self.done] = np.reshape(batch['done'], [self.batch_size, 1])
        fd[self.target_state] = target_state

        self.sess.run(self.update_all, feed_dict=fd)

        if self.epoch%10 == 0:
            summary = self.sess.run(self.merged, feed_dict=fd)
            self.writer.add_summary(summary, self.epoch)

        fd.clear()
        for key in self.observation_dim:
            fd[self.obs_in[key]] = \
                self.normalize_obs(batch[key+'_0'], key, self.batch_size)
        fd[self.action] = np.reshape(
            batch['action'], [self.batch_size]+self.action_dim)

        self.sess.run(self.update_actor, feed_dict=fd)

        self.sess.run(self.assign_target_soft)


    def reset(self):

        self.sess.run(self.var_init)

    
    def load(self, saved_variables):

        self.sess.run(
            [self.vars[var].assign(saved_variables[var]) for var in self.vars.keys()]
        )
        self.sess.run(self.assign_target)

    
    def return_variables(self):

        return {name:self.sess.run(name) for name in self.vars}


    def normalize_obs(self, in_, key, batch_size):

        in_ = np.array(in_, dtype=np.float32)

        if key =='lidar':
            out_ = np.exp(np.divide(-in_,np.float32(2.0)))
        elif key =='proximity':
            out_ = np.exp(-in_)
        elif key =='control':
            out_ = np.divide(in_, np.float32(self.a_scale))
        elif key =='depth':
            out_ = np.divide(in_, np.float32(255.0))
        elif key =='goal':
            out_ = np.exp(-in_)

        out_ = np.reshape(
            out_,
            [batch_size]+self.observation_dim[key]
        )

        return out_



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
