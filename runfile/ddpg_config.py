class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.load_buffer=True
        self.load_weight=True
        self.gpu=True
        self.state_dim=[43]
        self.action_dim=[3]
        self.action_max=[0.2,0.2,0.174]
        self.action_min=[-0.2,-0.2,-0.174]
        self.gamma=0.9 # discount factor
        self.critic_learning_rate=1e-3
        self.actor_learning_rate=1e-4
        self.tau=1e-3
        self.l2_penalty=1e-5
        self.max_buffer=1e+5
        self.batch_size=64
        self.max_step=1e+3
        self.max_episode=1e+4
        self.max_epoch=1e+7
        self.networks={
            'actor':[
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[self.state_dim[0],108]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[108,54]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[54,18]
                }
            ],
            'critic':[
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[self.state_dim[0]+self.action_dim[0],108]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[108,54]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[54,18]
                },
                {
                    'type':'dense',
                    'activation':'None',
                    'shape':[18,1]
                }
            ]
        }

config=Settings()
