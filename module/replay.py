from numpy import random

class Replay(object):
    
    def __init__(self,max_buffer,batch_size):
        self.max_buffer=max_buffer
        self.batch_size=batch_size
        self.currentPosition=-1
        self.buffer={
            'lidar0':[],
            'sonar0':[],
            'rgb0':[],
            'depth0':[],
            'lidar1':[],
            'sonar1':[],
            'rgb1':[],
            'depth1':[],
            'action0':[],
            'reward':[],
            'done':[]
        }
        self.buffersize=0
        self.max=False

    def batch(self):
        if self.buffersize>self.batch_size:
            indices=random.choice(range(self.buffersize),self.batch_size,replace=False)
            Batch={
                'lidar0':[],
                'sonar0':[],
                'rgb0':[],
                'depth0':[],
                'lidar1':[],
                'sonar1':[],
                'rgb1':[],
                'depth1':[],
                'action0':[],
                'reward':[],
                'done':[]
            }
            for name in self.buffer.keys():
                for idx in indices:
                    Batch[name].append(self.buffer[name][idx])
            return Batch
        else:
            return self.buffer

    def add(self,experience):
        if (self.currentPosition>=self.max_buffer-1):
            self.currentPosition=0
            self.max=True
        if self.max:
            for name in self.buffer.keys():
                self.buffer[name][self.currentPosition]=experience[name]
        else:
            for name in self.buffer.keys():
                self.buffer[name].append(experience[name])
            self.buffersize+=1
        self.currentPosition+=1
    
    def clear(self):
        self.currentPosition=-1
        self.buffersize=0
        self.buffer={
            'lidar0':[],
            'sonar0':[],
            'rgb0':[],
            'depth0':[],
            'lidar1':[],
            'sonar1':[],
            'rgb1':[],
            'depth1':[],
            'action0':[],
            'reward':[],
            'done':[]
        }
        self.max=False