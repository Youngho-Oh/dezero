import math
import random
import numpy as np

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True) :
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(self.dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)

        self.reset()
    
    def reset(self) :
        self.iteration = 0
        if self.shuffle :
            self.index = np.random.permutation(self.data_size)
        else :
            self.index = np.arange(self.data_size)
    
    def __iter__(self) :
        return self
    
    def __next__(self) :
        if self.iteration >= self.max_iter :
            self.reset()
            raise StopIteration
        
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([temp[0] for temp in batch])
        t = np.array([temp[1] for temp in batch])

        self.iteration += 1
        return x, t
    
    def next(self) :
        return self.__next__()