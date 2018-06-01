import tensorflow as tf
from agent_modules.build_actor_critic import Build_network
from agent_modules.additional_functions import l2_regularizer,gradient_inverter


class DDPG(object):

    def __init__(self,config):
        sess_config=tf.ConfigProto()
        sess_config.gpu_options.allow_growth=True
        self.gamma=tf.constant(config.gamma,dtype=tf.float32,name='gamma')
        self.sess=tf.Session(config=sess_config)
        self.var_init=tf.global_variables_initializer()
        self.reward=tf.placeholder(tf.float32,[None,1])
        self.done=tf.placeholder(tf.float32,[None,1])
        self.target_q=tf.placeholder(tf.float32,[None,1])
        # build network
        self.actor_net=Build_network(self.sess,config,'actor_net')
        self.actor_target=Build_network(self.sess,config,'actor_target')
        self.critic_net=Build_network(self.sess,config,'critic_net')
        self.critic_target=Build_network(self.sess,config,'critic_target')
        self.var_init=tf.global_variables_initializer()
        # update critic
        y=self.reward+tf.multiply(self.gamma,tf.multiply(self.target_q,self.done))
        q_loss=tf.reduce_sum(tf.pow(self.critic_net.out_-y,2))/config.batch_size+ \
            config.l2_penalty*l2_regularizer(self.critic_net.var_list)
        self.update_critic=tf.train.AdamOptimizer( \
            learning_rate=config.critic_learning_rate).minimize(q_loss)
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
        self.sess.run(self.var_init)
        self.sess.run(self.assign_target)

    def policy(self,state):
        return self.actor_net.evaluate(state)
    
    def reset(self):
        self.sess.run(self.var_init)

    def update(self,batch):
        target_action=self.actor_target.evaluate(batch['state1'])
        target_q=self.critic_target.evaluate(batch['state1'],target_action)
        self.sess.run(self.update_critic, \
                      feed_dict={self.critic_net.state:batch['state0'], \
                                 self.critic_net.action:batch['action0'], \
                                 self.reward:batch['reward'], \
                                 self.target_q:target_q, \
                                 self.done:batch['done']})
        self.sess.run(self.update_actor, \
                      feed_dict={self.critic_net.state:batch['state0'], \
                                 self.critic_net.action:batch['action0'], \
                                 self.actor_net.state:batch['state0']})
        self.sess.run(self.assign_target_soft)
    
    def load(self,saved_variables):
        self.sess.run( \
            [self.actor_net.variables[var].assign(saved_variables[var]) \
                for var in self.actor_net.variables.keys()]+ \
            [self.critic_net.variables[var].assign(saved_variables[var]) \
                for var in self.critic_net.variables.keys()]+ \
            self.assign_target)