from numpy import random

class Replay(object):
    
    def __init__(self,max_buffer,batch_size,observations=False):
        self.max_buffer=max_buffer
        self.batch_size=batch_size
        self.currentPosition=-1
        if not observations:
            observations=['vector','rgbd']
        self.bufferkeys=[key+'_0' for key in observations]+ \
                        [key+'_1' for key in observations]+ \
                        ['action','reward','done']
        self.buffer={key:[] for key in self.bufferkeys}
        self.buffersize=0
        self.max=False

    def batch(self):
        if self.buffersize>self.batch_size:
            indices=random.choice(range(self.buffersize),self.batch_size,replace=False)
            Batch={key:[] for key in self.buffer.keys()}
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
        self.buffer={key:[] for key in self.bufferkeys}
        self.max=False
