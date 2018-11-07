from numpy import random

class Replay(object):
    
    def __init__(self, max_buffer, batch_size, observations=False):
        self.max_buffer = max_buffer
        self.batch_size = batch_size
        self.currentPosition = -1
        if not observations:
            observations=['state']
        self.bufferkeys=[item+'_0' for item in observations]+ \
                        [item+'_1' for item in observations]+ \
                        ['action','reward','done']
        self.buffer = {item:[] for item in self.bufferkeys}
        self.buffersize = 0

    def batch(self):
        if self.buffersize > self.batch_size:
            indices = random.choice(
                range(self.buffersize),
                self.batch_size,
                replace=False
            )
            Batch = {key:[] for key in self.bufferkeys}
            for name in self.bufferkeys:
                for idx in indices:
                    Batch[name].append(self.buffer[name][idx])
            return Batch
        else:
            return self.buffer

    def add(self, experience):
        self.currentPosition += 1
        if self.buffersize >= self.max_buffer:
            if self.currentPosition > self.max_buffer-1:
                self.currentPosition = 0
            for item in self.bufferkeys:
                self.buffer[item][self.currentPosition] = experience[item]
        else:
            for item in self.bufferkeys:
                self.buffer[item].append(experience[item])
            self.buffersize += 1
        self.buffer['buffersize'] = self.buffersize
        self.buffer['currentPosition'] = self.currentPosition
    
    def load(self, buffer):
        self.buffersize = max(buffer['buffersize'], self.max_buffer)
        idx = buffer['currentPosition']+1
        if self.currentPosition > self.max_buffer-1:
            self.currentPosition = self.max_buffer-1
            for key in self.bufferkeys:
                self.buffer[key] = buffer[key][idx-self.max_buffer:idx]
        else:
            self.currentPosition = buffer['currentPosition']
            for key in self.bufferkeys:
                self.buffer[key] = buffer[key][0:idx]+buffer[key][-self.max_buffer+idx:]

    def clear(self):
        self.currentPosition = -1
        self.buffersize = 0
        self.buffer = {item:[] for item in self.bufferkeys}
