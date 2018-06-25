class Settings(object):
    
    def __init__(self):
        self.default()

    def default(self):
        self.gpu=True
        self.vector_dim=[None,10]
        self.rgbd_dim=[None,480,640,7]
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
            'vector':[
                [10,200],
                [200,200]
            ],
            'rgbd':[
                [7,7,7,8],
                [5,5,8,9],
                [3,3,9,10],
                [3,3,10,11],
                [3,3,11,12]
            ],
            'merge':[
                [-1,200],
                [200,100]
            ]
        }

config=Settings()