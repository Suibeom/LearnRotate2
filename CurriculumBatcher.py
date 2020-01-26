import itertools

import numpy as np



class BatchCombinedIterator():

    def __init__(self, a, b):
        self.a = a
        self.b = b
        a0, a1 = next(self.a)
        b0, b1 = next(self.b)
        self.current = np.concatenate((a0, b0)), np.concatenate((a1, b1))

    def __iter__(self):
        return self

    def __next__(self):
        a0, a1 = next(self.a)
        b0, b1 = next(self.b)
        self.current = np.concatenate((a0, b0)), np.concatenate((a1, b1))
        return self.current
