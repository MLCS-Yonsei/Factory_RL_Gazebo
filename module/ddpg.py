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
        self.epsilon=0.4
        self.action_dim=config.action_dim
        self.range_dim=copy.copy(config.range_dim)
        self.range_dim[0]=-1
        self.sonar_dim=copy.copy(config.sonar_dim)
        self.sonar_dim[0]=-1
        self.rgb_dim=copy.copy(config.rgb_dim)
        self.rgb_dim[0]=-1
        self.depth_dim=copy.copy(config.depth_dim)
        self.depth_dim[0]=-1
        self.gamma=tf.constant(config.gamma,dtype=tf.float32,name='gamma')
        self.sess=tf.Session(config=sess_config)
        self.var_init=tf.global_variables_initializer()
        self.reward=tf.placeholder(tf.float32,[None,1])
        self.done=tf.placeholder(tf.float32,[None,1])
        self.target_q=tf.placeholder(tf.float32,[None,1])
        # self.noise=tf.placeholder(tf.float32,[None,config.action_dim])
        # build network
        self.actor_net=Build_network(self.sess,config,'actor_net')
        self.actor_target=Build_network(self.sess,config,'actor_target')
        self.critic_net=Build_network(self.sess,config,'critic_net')
        self.critic_target=Build_network(self.sess,config,'critic_target')
        # update critic
        y=self.reward+tf.multiply(self.gamma,tf.multiply(self.target_q,1.0-self.done))
        # y=self.reward+tf.multiply(self.gamma,self.target_q)
        q_loss=tf.reduce_sum(tf.pow(self.critic_net.out_-y,2))/config.batch_size+ \
            config.l2_penalty*l2_regularizer(self.critic_net.var_list)
        self.update_critic=tf.train.AdamOptimizer( \
            learning_rate=config.critic_learning_rate).minimize(q_loss,var_list=self.critic_net.var_list)
        # update actor
        act_grad_v=tf.gradients(self.critic_net.out_,self.critic_net.action)
        action_gradients=[act_grad_v[0]/tf.to_float(tf.shape(act_grad_v[0])[0])]
        del_Q_a=gradient_inverter( \
            config.action_bounds,action_gradients,self.actor_net.out_)
        parameters_gradients=tf.gradients(
            self.actor_net.out_,self.actor_net.var_list,-del_Q_a)
        self.update_actor=tf.train.AdamOptimizer( \
            learning_rate=config.actor_learning_rate) \
            .apply_gradients(zip(parameters_gradients,self.actor_net.var_list))
        # target copy
        self.assign_target= \
            [self.actor_target.variables[var].assign( \
                self.actor_net.variables[var.replace('_target','_net')] \
            ) for var in self.actor_target.variables.keys()]+ \
            [self.critic_target.variables[var].assign( \
                self.critic_net.variables[var.replace('_target','_net')] \
            ) for var in self.critic_target.variables.keys()]
        self.assign_target_soft= \
            [self.actor_target.variables[var].assign( \
                config.tau*self.actor_net.variables[var.replace('_target','_net')]+ \
                (1-config.tau)*self.actor_target.variables[var] \
            ) for var in self.actor_target.variables.keys()]+ \
            [self.critic_target.variables[var].assign( \
                config.tau*self.critic_net.variables[var.replace('_target','_net')]+ \
                (1-config.tau)*self.critic_target.variables[var] \
            ) for var in self.critic_target.variables.keys()]
        # initialize variables
        self.var_init=tf.global_variables_initializer()
        self.sess.run(self.var_init)
        self.sess.run(self.assign_target)
        self.a_scale,self.a_mean=self.sess.run(
            [self.actor_net.a_scale,self.actor_net.a_mean])

    def chooseAction(self,lidar,sonar,rgb,depth):
        action=self.sess.run(self.actor_net.out_before_activation, \
            feed_dict={self.actor_net.state_lidar:lidar, \
                       self.actor_net.state_sonar:sonar, \
                       self.actor_net.state_rgb:rgb, \
                       self.actor_net.state_depth:depth})
        action=self.a_scale* \
               np.tanh(action+self.epsilon*np.random.randn(1,self.action_dim))+ \
               self.a_mean
        return action

    def learn(self, batch):
        lidar0=np.reshape(batch['lidar0'],self.range_dim)
        sonar0=np.reshape(batch['sonar0'],self.sonar_dim)
        rgb0=np.reshape(batch['rgb0'],self.rgb_dim)
        depth0=np.reshape(batch['depth0'],self.depth_dim)
        lidar1=np.reshape(batch['lidar1'],self.range_dim)
        sonar1=np.reshape(batch['sonar1'],self.sonar_dim)
        rgb1=np.reshape(batch['rgb1'],self.rgb_dim)
        depth1=np.reshape(batch['depth1'],self.depth_dim)
        action0=np.reshape(batch['action0'],[-1,self.action_dim])
        reward=np.reshape(batch['reward'],[-1,1])
        done=np.reshape(batch['done'],[-1,1])
        target_action=self.actor_target.evaluate(lidar1,sonar1,rgb1,depth1)
        target_q=self.critic_target.evaluate(lidar1,sonar1,rgb1,depth1,action=target_action)
        self.sess.run(self.update_critic, \
                      feed_dict={self.critic_net.state_lidar:lidar0, \
                                 self.critic_net.state_sonar:sonar0, \
                                 self.critic_net.state_rgb:rgb0, \
                                 self.critic_net.state_depth:depth0, \
                                 self.critic_net.action:action0, \
                                 self.reward:reward, \
                                 self.target_q:target_q, \
                                 self.done:done})
        self.sess.run(self.update_actor, \
                      feed_dict={self.critic_net.state_lidar:lidar0, \
                                 self.critic_net.state_sonar:sonar0, \
                                 self.critic_net.state_rgb:rgb0, \
                                 self.critic_net.state_depth:depth0, \
                                 self.critic_net.action:action0, \
                                 self.actor_net.state_lidar:lidar0, \
                                 self.actor_net.state_sonar:sonar0, \
                                 self.actor_net.state_rgb:rgb0, \
                                 self.actor_net.state_depth:depth0})
        self.sess.run(self.assign_target_soft)

    def reset(self):
        self.sess.run(self.var_init)
    
    def load(self,saved_variables):
        self.sess.run( \
            [self.actor_net.variables[var].assign(saved_variables[var]) \
                for var in self.actor_net.variables.keys()]+ \
            [self.critic_net.variables[var].assign(saved_variables[var]) \
                for var in self.critic_net.variables.keys()]+ \
            self.assign_target)
    
    def return_variables(self):
        return dict({name:self.sess.run(name) \
                    for name in self.actor_net.variables.keys()}, \
               **{name:self.sess.run(name) \
                    for name in self.critic_net.variables.keys()})

class Build_network(object):

    def __init__(self,sess,config,name):
        self.name=name
        self.sess=sess
        layers=config.layers
        self.trainable=False if name.split('_')[1]=='target' else True
        with tf.name_scope(name):
            self.state_lidar=tf.placeholder(tf.float32,config.range_dim)
            self.state_sonar=tf.placeholder(tf.float32,config.sonar_dim)
            self.state_rgb=tf.placeholder(tf.float32,config.rgb_dim)
            self.state_depth=tf.placeholder(tf.float32,config.depth_dim)
            layers['output']=[[layers['merge'][-1][-1],1]]
            if name[0]=='a':
                # config.action_dim=len(config.action_bounds[0])
                self.a_scale=tf.subtract(
                    config.action_bounds[0],config.action_bounds[1])/2.0
                self.a_mean=tf.add(
                    config.action_bounds[0],config.action_bounds[1])/2.0
                layers['output'][0][1]=config.action_dim
            else:
                layers['merge'][0][0]+=config.action_dim
                self.action=tf.placeholder(tf.float32,[None,config.action_dim])
            for item in layers.keys():
                for idx,shape in enumerate(layers[item]):
                    self.create_variable(shape,item+str(idx))
            self.var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,name)
            self.variables={var.name:var for var in self.var_list}
            out_lidar=self.state_lidar
            for layer in range(layers['lidar']):
                out_lidar=self.fc(out_lidar,'lidar'+str(layer))
            out_sonar=self.state_sonar
            for layer in range(layers['sonar']):
                out_sonar=self.fc(out_sonar,'sonar'+str(layer))
            out_rgb=self.state_rgb
            for layer in range(layers['rgb']):
                out_rgb=self.conv(out_rgb,'rgb'+str(layer))
            out_rgb=tf.reshape(out_rgb, \
                [
                    -1,
                    layers['rgb'][-1][-1]* \
                    config.rgb_dim[1]* \
                    config.rgb_dim[2]/ \
                    2**(2*len(layers['rgb']))
                ])
            out_depth=self.state_depth
            for layer in range(layers['depth']):
                out_depth=self.conv(out_depth,'depth'+str(layer))
            out_depth=tf.reshape(out_depth, \
                [
                    -1,
                    layers['depth'][-1][-1]* \
                    config.depth_dim[1]* \
                    config.depth_dim[2]/ \
                    2**(2*len(layers['depth']))
                ])
            out_=tf.concat([out_lidar,out_sonar,out_rgb,out_depth],1) \
                if name[0]=='c' else \
                tf.concat([out_lidar,out_sonar,out_rgb,out_depth,self.action],1)
            for layer in range(layers['merge']):
                out_=self.fc(out_,'fc'+str(layer))
            if name[0]=='a':
                self.out_before_activation=out_
                self.out_=tf.multiply(tf.tanh(out_),self.a_scale)+self.a_mean
            else:
                self.out_=out_

    def evaluate(self,lidar,sonar,rgb,depth,action=None):
        feed_dict={
                self.state_lidar:lidar,
                self.state_sonar:sonar,
                self.state_rgb:rgb,
                self.state_depth:depth}
        if self.name[0]=='c':
            feed_dict[self.action]=action
        return self.sess.run(self.out_,feed_dict=feed_dict)    

    def fc(self,in_,layer):
        return tf.nn.relu(tf.matmul(in_, \
            self.variables[self.name+'/'+layer+'/w:0'])+ \
            self.variables[self.name+'/'+layer+'/b:0'])
    
    def conv(self,in_,layer):
        return tf.nn.max_pool(
            tf.nn.relu(
                tf.nn.conv2d(
                    in_,
                    self.variables[self.name+'/'+layer+'/f:0'],
                    strides=[1,1,1,1],
                    padding='SAME')),
            ksize=[1,2,2,1],
            strides=[1,2,2,1],
            padding='SAME')

    def create_variable(self,shape,name):
        with tf.name_scope(name):
            if len(shape)==2:
                stddev=1/np.sqrt(shape[0])
                tf.Variable( \
                    tf.random_normal(shape,stddev=stddev),name='w',trainable=self.trainable)
                tf.Variable( \
                    tf.random_normal(shape[1],stddev=stddev),name='b',trainable=self.trainable)
            else:
                stddev=1/np.sqrt(shape[0]*shape[1]*shape[2])
                tf.Variable( \
                    tf.random_normal(shape,stddev=stddev),name='f',trainable=self.trainable)


def l2_regularizer(vars):
    loss=0
    for var in vars:
        loss+=tf.reduce_sum(tf.pow(var,2))
    return loss/2.0

def gradient_inverter(action_bounds,action_gradients,actions):
    action_dim=len(action_bounds[0])
    pmax=tf.constant(action_bounds[0],dtype=tf.float32)
    pmin=tf.constant(action_bounds[1],dtype=tf.float32)
    plidar=tf.constant([x-y for x,y in zip(action_bounds[0],action_bounds[1])],dtype=tf.float32)
    pdiff_max=tf.div(-actions+pmax,plidar)
    pdiff_min=tf.div(actions-pmin,plidar)
    zeros_act_grad_filter=tf.zeros([action_dim])       
    return tf.where( \
        tf.greater(action_gradients,zeros_act_grad_filter), \
        tf.multiply(action_gradients,pdiff_max), \
        tf.multiply(action_gradients,pdiff_min))
