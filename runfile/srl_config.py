class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.load_buffer=True
        self.gpu=True

        # learning parameters
        self.gamma=0.99 # discount factor
        self.critic_learning_rate=1e-3
        self.actor_learning_rate=1e-4
        self.srl_learning_rate=1e-3
        self.tau=1e-3
        self.l2_penalty=1e-5
        self.max_buffer=1e+5
        self.batch_size=4
        self.max_step=1e+3
        self.max_episode=1e+4
        self.max_epoch=1e+7
        
        # dimension setup
        self.observation_dim={
            'lidar':[36,1,3],
            'proximity':[4],
            'control':[3],
            'depth':[96,128,3],
            'goal':[2]
        }
        self.state_dim=[36]
        self.action_dim=[3]        
        self.action_max=[0.1,0.1,0.2]
        self.action_min=[-0.1,-0.1,-0.2]

        # loss
        self.c_srl = 0.25
        self.c_rew = 0.5
        self.c_slow = 1.0
        self.c_div = 1.0
        self.c_inv = 0.5
        self.c_fwd = 1.0


        # layer setup
        self.observation_networks={
            'lidar':[
                {
                    'type':'conv',
                    'activation':'prelu',
                    'shape':[5,1,self.observation_dim['lidar'][2],6],
                    'strides':[1,1,1,1],
                    'pool':[1,3,1,1]
                },
                {
                    'type':'conv',
                    'activation':'prelu',
                    'shape':[3,1,6,9],
                    'strides':[1,1,1,1],
                    'pool':[1,3,1,1]
                },
                {
                    'type':'flatten',
                    'activation':'softplus',
                    'shape':[-1,self.state_dim[0]]
                }
            ],
            'proximity':[
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[self.observation_dim['proximity'][0],12]
                },
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[12,24]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[24,self.state_dim[0]]
                }
            ],
            'control':[
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[self.observation_dim['control'][0],12]
                },
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[12,24]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[24,self.state_dim[0]]
                }
            ],
            'depth':[
                {
                    'type':'conv',
                    'activation':'prelu',
                    'shape':[7,7,self.observation_dim['depth'][2],6],
                    'strides':[1,1,1,1],
                    'pool':[1,2,2,1]
                },
                {
                    'type':'conv',
                    'activation':'prelu',
                    'shape':[5,5,6,9],
                    'strides':[1,1,1,1],
                    'pool':[1,2,2,1]
                },
                {
                    'type':'conv',
                    'activation':'prelu',
                    'shape':[3,3,9,12],
                    'strides':[1,1,1,1],
                    'pool':[1,2,2,1]
                },
                {
                    'type':'conv',
                    'activation':'prelu',
                    'shape':[3,3,12,15],
                    'strides':[1,1,1,1],
                    'pool':[1,2,2,1]
                },
                {
                    'type':'conv',
                    'activation':'prelu',
                    'shape':[3,3,15,18],
                    'strides':[1,1,1,1],
                    'pool':[1,2,2,1]
                },
                {
                    'type':'flatten',
                    'activation':'prelu',
                    'shape':[-1,108]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[108,self.state_dim[0]]
                }
            ],
            'goal':[
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[self.observation_dim['goal'][0],12]
                },
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[12,24]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[24,self.state_dim[0]]
                }
            ]
        }
        self.prediction_networks={
            'reward':[
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[self.state_dim[0]+self.action_dim[0],64]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[64,32]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[32,16]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[16,1]
                }
            ],
            'state':[
                {
                    'type':'dense',
                    'activation':'prelu',
                    'shape':[self.state_dim[0]+self.action_dim[0],72]
                },
                {
                    'type':'dense',
                    'activation':'softplus',
                    'shape':[72,108]
                },
                {
                    'type':'dense',
                    'activation':'None',
                    'shape':[108,self.state_dim[0]]
                }
            ]
        }
        self.rl_networks={
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
