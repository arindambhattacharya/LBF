import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
from random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import dpbf


def mhash(inp):
    return abs(hash(inp.tostring()))


class dpbf_logistic:
    def __init__(self, model):
        self.fprob = 0.01
        self.model = model
        # self.tao = tao

    def initialize(self, x, y, n=128, p=0.01):
        fprob = p
        k = 3
        counter_chunk = 1
        num_partition = 30
        dpbf.init_bloom_filter(n, k, counter_chunk, num_partition, fprob)
        for i in range(len(x)):
            if np.allclose(y[i], np.array([1, 0])): #y = [p_true, p_false]
                self.insert(x[i])
        dpbf.update()

    def insert(self, idx):
        y = self.model.predict([idx]) #predict returns [p_true, p_false]
        if np.argmax(y) == 1:
            dpbf.insert(mhash(idx))

    def check(self, idx):
        return np.argmax(self.model.predict([x])) == '0' or dpbf.check(mhash(idx))

    def get_fpr(self, X, Y):
        fp = len([x for x, y in zip(X, Y) if self.check(x) and np.allclose(y, np.array([0, 1]))])
        n = len([y for y in Y if np.allclose(y, np.array([0, 1]))])
        return fp / n

    def get_size(self):
        return dpbf.getMemory()


# x_train = []
# y_train = []

# # X = X.astype('float64').round(decimals = 6)
# # y = y.astype('float64').round(decimals = 6)

# for i in range(1,1000):
#     x_train.append([i,i])
#     y_train.append(i %2)

# dpbf = dpbf_logistic(0.0001,0.51)
# dpbf.initialize(x_train,y_train)
# print(dpbf.get_fpr(x_train,y_train))

# X_train_full = np.genfromtxt('X_train.bin', delimiter=' ')
# Y_train_full = np.genfromtxt('Y_train.bin', delimiter=' ')

# X_test_full = np.genfromtxt('X_test.bin', delimiter=' ')
# Y_test_full = np.genfromtxt('Y_test.bin', delimiter=' ')

# res = []
# sizes = [1000, 10000, 100000, 1000000]
# for s in sizes:
#     X_train = X_train_full[:s]
#     Y_train = Y_train_full[:s]
#     my_bc = dpbf_logistic(0.0001,0.5)
#     c = my_bc.initialize(X_train, Y_train)
#     res.append(c)
