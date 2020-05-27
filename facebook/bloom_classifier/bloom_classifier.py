import numpy as np
import copy
import sys
import logging
import bloom_filter as bf

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class BloomClassifier(object):
    """
    Bloom Classifier
    @param model: A pretrained initial classifier.
    """

    def __init__(self, model):
        super(BloomClassifier, self).__init__()
        self.models = [model]
        self.count = 0

    def initialize(self, x, y, n=None, m=None, k=3, p=None):
        self.overflow_filter = bf.BloomFilter(m, n, k, p)
        for i in range(len(x)):
            if y[i]:
                self.insert_one(x[i])

    def insert(self, X):
        for x in X:
            self.insert_one(x)

    def insert_one(self, x):
        for model in self.models:
            if model.predict([x]):
                break
            self.overflow_filter.insert(x)

    def add_data(self, x, y, model=None):
        if model:
            self.models.append(model)
        for i in range(len(x)):
            if y[i]:
                self.insert_one(x[i])

    def check(self, x):
        for model in self.models:
            if model.predict([x]):
                return True
        return self.overflow_filter.check(x)

    def get_fpr(self, X, Y):
        fp = len([x for x, y in zip(X, Y) if self.check(x) and y == 0])
        n = len([y for y in Y if y == 0])
        return fp / n

    def get_clf_size(self):
        return sys.getsizeof(self.models)
    
    def get_bf_size(self):
        return sys.getsizeof(self.overflow_filter)
    
    def get_size(self):
        return sys.getsizeof(self.models) + sys.getsizeof(self.overflow_filter)
        # return sys.getsizeof(self.overflow_filter)
