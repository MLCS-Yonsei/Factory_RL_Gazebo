class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.load_buffer=True
        self.gpu=True
        
        # dimension setup
        self.range_dim=(36)
        self.proximity_dim=(4)
        self.control_dim=(3)
        self.rgbd_dim=(96,128,7)
        self.state_dim=(50)
        self.action_dim=(3)
        
        self.action_bounds=[[0.2,0.2,0.5],[-0.2,-0.2,-0.5]] # [max,min]
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
            'range':[
                ('fc',self.range_dim[0],200),
                ('fc',200,self.state_dim[0])
            ],
            'proximity':[
               ('fc',self.proximity_dim[0],200),
                ('fc',200,self.state_dim[0])
            ],
            'control':[
                ('fc',self.control_dim[0],200),
                ('fc',200,self.state_dim[0])
            ],
            'rgbd':[
                ('conv',5,5,7,8),
                ('conv',3,3,8,9),
                ('conv',3,3,9,10)
            ],
            'reward_prediction':[
                ('fc',self.state_dim[0],60),
                ('fc',40,30)
            ],
            'state_prediction':[
                ('fc',self.state_dim[0],100),
                ('fc',100,80)
            ],
            'actor':[
                ('fc',self.state_dim[0],40),
                ('fc',40,30)
            ],
            'critic':[
                ('fc',self.state_dim[0],40),
                ('fc',40+self.action_dim[0],30)
            ]
        }

config=Settings()
