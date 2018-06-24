class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.gpu=True
        self.range_dim=[None,10]
        self.sonar_dim=[None,4]
        self.rgb_dim=[None,480,640,3]
        self.depth_dim=[None,320,480,1]
        self.action_dim=3
        self.action_bounds=[[0.5,0.5,0.5],[-0.5,-0.5,-0.5]] # [max,min]
        self.gamma=0.9 # discount factor
        self.critic_learning_rate=1e-3
        self.actor_learning_rate=1e-4
        self.tau=1e-3
        self.l2_penalty=1e-5
        self.max_buffer=1e+5
        self.batch_size=1e+3
        self.max_step=1e+3
        self.max_episode=1e+4
        self.max_epoch=1e+7
        self.layers={
            'range':[
                [10,200],
                [200,200]
            ],
            'sonar':[
                [10,200],
                [200,200]
            ],
            'rgb':[
                [5,5,3,8],
                [3,3,6,16]
            ],
            'depth':[
                [5,5,1,4],
                [3,3,2,8]
            ],
            'merge':[
                [500,200],
                [200,100]
            ]
        }

config=Settings()