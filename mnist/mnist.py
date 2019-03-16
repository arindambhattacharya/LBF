import tflearn
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.model_selection import train_test_split
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge

import numpy as np
import pickle
import sys
import copy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")  # shouldn't do this

tflearn.config.init_graph(
    seed=None,
    log_device=False,
    num_cores=6,
    gpu_memory_fraction=0.75,
    soft_placement=True)


def get_proper_images(raw):
    raw_float = np.array(raw, dtype=float)
    images = raw_float.reshape([-1, 3, 32, 32])
    images = images.transpose([0, 2, 3, 1])
    return images


def onehot_labels(labels):
    return np.eye(2)[labels]


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def filter_classes(X, Y, classes):
    # d = [(x, y) for x, y in zip(X, Y) if y in classes]
    # return zip(*d)
    X_ = []
    Y_ = []
    for x, y in zip(X, Y):
        if y in classes:
            X_.append(x)
            Y_.append(y)
    return X_, Y_


def binarize_classes(X, Y, thr):
    X_ = []
    Y_ = []
    for x, y in zip(X, Y):
        if y > thr:
            y = 1
        elif y <= thr:
            y = 0
        X_.append(x)
        Y_.append(y)
    return X_, Y_


########################################################################################
# Load and prepare data

import tflearn.datasets.mnist as mnist
X, Y, testX, testY = mnist.load_data(one_hot=False)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

X, Y = binarize_classes(X, Y, 4)

# X, Y = filter_classes(X, Y, [0, 1])

Y = onehot_labels(Y)

X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)

X_init, X_insert, Y_init, Y_insert = train_test_split(X, Y, test_size=0.5)

X_test_init, X_test_insert, Y_test_init, Y_test_insert = train_test_split(
    X_test, Y_test, test_size=0.5)

##################################################################################################
# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(
    network,
    optimizer='adam',
    learning_rate=0.01,
    loss='categorical_crossentropy',
    name='target')
model = tflearn.DNN(
    network, tensorboard_verbose=1, tensorboard_dir='summaries')

##################################################################################################
# Train using classifier. Done once
# model.fit(
#     X_init, Y_init, n_epoch=20, validation_set=(X_test_init, Y_test_init),
#     show_metric=True, run_id="MNIST_binary")

# model.save('models/mnist.model')

##################################################################################################
# IA-LBF experiments

import time
sys.path.append('dpbf_classifier')
import dpbf_classification as dc

model.load('models/mnist.model')
my_dc = dc.dpbf_logistic(model)

model_size = sys.getsizeof(model)
model_size_uncompressed = sys.getsizeof(tflearn.variables.get_all_variables())

outfile = open('outputs/mnist_ialbf_hi_run1.txt', 'w')

start = time.time()

my_dc.initialize(X_init, Y_init, p=0.08)

log = 'DPBF initialization time: {0:.2f} seconds\n'.format(time.time() - start)
print(log)
outfile.write(log)

log = 'Initial false positive on train data: {0:.6f}\n'.format(my_dc.get_fpr(X_init, Y_init))
print(log)
outfile.write(log)

log = 'Initial memory (Bloom filter): {0:.2f} bytes\n'.format(my_dc.get_size())
print(log)
outfile.write(log)

log = 'Initial memory (Classifier, compressed): {0:.2f} bytes\n'.format(model_size)
print(log)
outfile.write(log)

log = 'Initial memory (Classifier, uncompressed): {0:.2f} bytes\n'.format(model_size_uncompressed)
print(log)
outfile.write(log)

log = 'Initial total memory: {0:.2f} bytes\n'.format(model_size + my_dc.get_size())
print(log)
outfile.write(log)

start = time.time()

for x, y in zip(X_insert, Y_insert):
    if np.allclose(y, np.array([1, 0])):
        my_dc.insert(x)

log = 'Average insertion time per 1000 elements: {0:.2f} seconds\n'.format((time.time() - start) * 1000 / len(X_insert))
print(log)
outfile.write(log)

log = 'False positive after insertion (inserted data): {0:.6f}\n'.format(my_dc.get_fpr(X_insert, Y_insert))
print(log)
outfile.write(log)

log = 'False positive after insertion (test data): {0:.6f}\n'.format(my_dc.get_fpr(X_test, Y_test))
print(log)
outfile.write(log)

log = 'False positive after insertion (entire data): {0:.6f}\n'.format(my_dc.get_fpr(np.concatenate((X, X_test)), np.concatenate((Y, Y_test))))
print(log)
outfile.write(log)

log = 'Memory after insertion (Bloom filter): {0:.2f} bytes\n'.format(my_dc.get_size())
print(log)
outfile.write(log)

log = 'Total memory after insertion: {0:.2f} bytes'.format(model_size + my_dc.get_size())
print(log)
outfile.write(log)

outfile.close()

##################################################################################################
# CA-LBF I experiments
# import time
# sys.path.append('bloom_classifier')
# import bloom_classifier as bc

# model.load('models/mnist.model')
# my_bc = bc.BloomClassifier(model)

# model_size = sys.getsizeof(model)
# model_size_uncompressed = sys.getsizeof(tflearn.variables.get_all_variables())

# outfile = open('outputs/mnist_calbf1_run3.txt', 'w')

# start = time.time()

# my_bc.initialize(X_init, Y_init, m=2500)

# log = 'BF initialization time: {0:.2f} seconds\n'.format(time.time() - start)
# print(log)
# outfile.write(log)

# log = 'Initial false positive on train data: {0:.6f}\n'.format(my_bc.get_fpr(X_init, Y_init))
# print(log)
# outfile.write(log)

# log = 'Initial memory (Bloom filter): {0:.2f} bytes\n'.format(my_bc.get_size())
# print(log)
# outfile.write(log)

# log = 'Initial memory (Classifier, compressed): {0:.2f} bytes\n'.format(model_size)
# print(log)
# outfile.write(log)

# log = 'Initial memory (Classifier, uncompressed): {0:.2f} bytes\n'.format(model_size_uncompressed)
# print(log)
# outfile.write(log)

# log = 'Initial total memory: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
# print(log)
# outfile.write(log)
# start1 = time.time()

# model.fit(
#      X_insert, Y_insert, n_epoch=1, validation_set=(X_test_insert, Y_test_insert),
#      show_metric=True, run_id="MNIST_binary")

# model.save('models/mnist_calbf1.model')

# # model.load('models/mnist_calbf1.model')

# start2 = time.time()

# my_bc.add_data(X_insert, Y_insert, model)

# log = 'Average insertion time per 1000 elements: {0:.2f} seconds\n'.format((time.time() - start2) * 1000/ len(X_insert))
# print(log)
# outfile.write(log)

# log = 'Average insertion time per 1000 elements including training: {0:.2f} seconds\n'.format((time.time() - start1) * 1000/ len(X_insert))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (inserted data): {0:.6f}\n'.format(my_bc.get_fpr(X_insert, Y_insert))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (test data): {0:.6f}\n'.format(my_bc.get_fpr(X_test, Y_test))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (entire data): {0:.6f}\n'.format(my_bc.get_fpr(np.concatenate((X, X_test)), np.concatenate((Y, Y_test))))
# print(log)
# outfile.write(log)

# log = 'Memory after insertion (Bloom filter): {0:.2f} bytes\n'.format(my_bc.get_size())
# print(log)
# outfile.write(log)

# log = 'Total memory after insertion: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
# print(log)
# outfile.write(log)

# outfile.close()

##################################################################################################

# CA-LBF II experiments

# import time
# sys.path.append('bloom_classifier')
# import bloom_classifier as bc

# model.load('models/mnist.model')
# my_bc = bc.BloomClassifier(model)

# model_size = sys.getsizeof(model)
# model_size_uncompressed = sys.getsizeof(tflearn.variables.get_all_variables())

# outfile = open('outputs/mnist_calbf2_run3.txt', 'w')

# start = time.time()

# my_bc.initialize(X_init, Y_init, m=2500)

# log = 'BF initialization time: {0:.2f} seconds\n'.format(time.time() - start)
# print(log)
# outfile.write(log)

# log = 'Initial false positive on train data: {0:.6f}\n'.format(my_bc.get_fpr(X_init, Y_init))
# print(log)
# outfile.write(log)

# log = 'Initial memory (Bloom filter): {0:.2f} bytes\n'.format(my_bc.get_size())
# print(log)
# outfile.write(log)

# log = 'Initial memory (Classifier, compressed): {0:.2f} bytes\n'.format(model_size)
# print(log)
# outfile.write(log)

# log = 'Initial memory (Classifier, uncompressed): {0:.2f} bytes\n'.format(model_size_uncompressed)
# print(log)
# outfile.write(log)

# log = 'Initial total memory: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
# print(log)
# outfile.write(log)

# start1 = time.time()

# tf.reset_default_graph() # reset the model for calbf 2

# model.fit(
#      X_insert, Y_insert, n_epoch=10, validation_set=(X_test_insert, Y_test_insert),
#      show_metric=True, run_id="MNIST_binary")

# model.save('models/mnist_calbf2.model')

# # model.load('models/mnist_calbf2.model')

# start2 = time.time()

# my_bc.add_data(X_insert, Y_insert, model)

# log = 'Average insertion time per 1000 elements: {0:.2f} seconds\n'.format((time.time() - start2) * 1000/ len(X_insert))
# print(log)
# outfile.write(log)

# log = 'Average insertion time per 1000 elements including training: {0:.2f} seconds\n'.format((time.time() - start1) * 1000/ len(X_insert))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (inserted data): {0:.6f}\n'.format(my_bc.get_fpr(X_insert, Y_insert))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (test data): {0:.6f}\n'.format(my_bc.get_fpr(X_test, Y_test))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (entire data): {0:.6f}\n'.format(my_bc.get_fpr(np.concatenate((X, X_test)), np.concatenate((Y, Y_test))))
# print(log)
# outfile.write(log)

# log = 'Memory after insertion (Bloom filter): {0:.2f} bytes\n'.format(my_bc.get_size())
# print(log)
# outfile.write(log)

# log = 'Total memory after insertion: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
# print(log)
# outfile.write(log)

# outfile.close()


##################################################################################################

# Baseline experiments

# import time
# sys.path.append('bloom_classifier')
# import bloom_classifier as bc

# model.load('models/mnist.model')
# my_bc = bc.BloomClassifier(model)

# model_size = sys.getsizeof(model)
# model_size_uncompressed = sys.getsizeof(tflearn.variables.get_all_variables())

# outfile = open('outputs/mnist_base_run1.txt', 'w')

# start = time.time()

# my_bc.initialize(X_init, Y_init, m=2500)

# log = 'BF initialization time: {0:.2f} seconds\n'.format(time.time() - start)
# print(log)
# outfile.write(log)

# log = 'Initial false positive on train data: {0:.6f}\n'.format(my_bc.get_fpr(X_init, Y_init))
# print(log)
# outfile.write(log)

# log = 'Initial memory (Bloom filter): {0:.2f} bytes\n'.format(my_bc.get_size())
# print(log)
# outfile.write(log)

# log = 'Initial memory (Classifier, compressed): {0:.2f} bytes\n'.format(model_size)
# print(log)
# outfile.write(log)

# log = 'Initial memory (Classifier, uncompressed): {0:.2f} bytes\n'.format(model_size_uncompressed)
# print(log)
# outfile.write(log)

# log = 'Initial total memory: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
# print(log)
# outfile.write(log)

# start1 = time.time()
# start2 = time.time()

# my_bc.add_data(X_insert, Y_insert, model)

# log = 'Average insertion time per 1000 elements: {0:.2f} seconds\n'.format((time.time() - start2) * 1000/ len(X_insert))
# print(log)
# outfile.write(log)

# log = 'Average insertion time per 1000 elements including training: {0:.2f} seconds\n'.format((time.time() - start1) * 1000/ len(X_insert))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (inserted data): {0:.6f}\n'.format(my_bc.get_fpr(X_insert, Y_insert))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (test data): {0:.6f}\n'.format(my_bc.get_fpr(X_test, Y_test))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (entire data): {0:.6f}\n'.format(my_bc.get_fpr(np.concatenate((X, X_test)), np.concatenate((Y, Y_test))))
# print(log)
# outfile.write(log)

# log = 'Memory after insertion (Bloom filter): {0:.2f} bytes\n'.format(my_bc.get_size())
# print(log)
# outfile.write(log)

# log = 'Total memory after insertion: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
# print(log)
# outfile.write(log)

# outfile.close()


print('\007')
