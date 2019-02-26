import tflearn
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import batch_normalization
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.model_selection import train_test_split

import numpy as np
import pickle
import sys
import copy

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore") # shouldn't do this

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


########################################################################################
# Load and prepare data

data1 = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_1')
data2 = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_2')
data3 = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_3')
data4 = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_4')
data5 = unpickle('cifar-10-python/cifar-10-batches-py/data_batch_5')

X = np.concatenate((get_proper_images(data1[b'data']),
                    get_proper_images(data2[b'data']),
                    get_proper_images(data3[b'data']),
                    get_proper_images(data4[b'data']),
                    get_proper_images(data5[b'data'])))
Y = np.concatenate(((data1[b'labels']), (data2[b'labels']), (data3[b'labels']),
                    (data4[b'labels']), (data5[b'labels'])))

X_test = get_proper_images(
    unpickle('cifar-10-python/cifar-10-batches-py/test_batch')[b'data'])

Y_test = unpickle('cifar-10-python/cifar-10-batches-py/test_batch')[b'labels']

X, Y = filter_classes(X, Y, [0, 1])
Y = onehot_labels(Y)

X_init, X_insert, Y_init, Y_insert = train_test_split(X, Y, test_size=0.5)

X_test, Y_test = filter_classes(X_test, Y_test, [0, 1])
Y_test = onehot_labels(Y_test)

X_test_init, X_test_insert, Y_test_init, Y_test_insert = train_test_split(
    X_test, Y_test, test_size=0.5)

##################################################################################################
# Preprocessing

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=15.)
# img_aug.add_random_crop((28, 28))
#img_aug.add_random_blur(sigma_max=5.0)

##################################################################################################
# Building Residual Network

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5
net = tflearn.input_data(
    shape=[None, 32, 32, 3],
    data_preprocessing=img_prep,
    data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.0001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n - 1, 32)
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n - 1, 64)
net = tflearn.residual_block(net, n - 1, 64)  #try
net = tflearn.residual_block(net, n - 1, 64)  #try
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 10, activation='relu')  #try
net = tflearn.fully_connected(net, 2, activation='softmax')
mom = tflearn.Momentum(0.1, lr_decay=0.1, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom, loss='categorical_crossentropy')

model = tflearn.DNN(
    net,
    tensorboard_verbose=0,
    tensorboard_dir='summaries',
    checkpoint_path='models/cifar10',
    max_checkpoints=1,
    clip_gradients=0.)

##################################################################################################
# Train using classifier. Done once

# model.fit(
#     X_init,
#     Y_init,
#     n_epoch=100,
#     shuffle=True,
#     snapshot_epoch=False,
#     snapshot_step=500,
#     validation_set=(X_test_init, Y_test_init),
#     show_metric=True,
#     batch_size=128,
#     run_id="cifar10_res_binary")

# model.save('models/cifar10.model')

##################################################################################################
# IA-LBF experiments

# import time
# sys.path.append('dpbf_classifier')
# import dpbf_classification as dc


# model.load('models/cifar10.model')
# my_dc = dc.dpbf_logistic(model)

# model_size = sys.getsizeof(model)
# model_size_uncompressed = sys.getsizeof(tflearn.variables.get_all_variables())

# outfile = open('cifar_ialbf_run3.txt', 'w')

# start = time.time()

# my_dc.initialize(X_init, Y_init)

# log = 'DPBF initialization time: {0:.2f} seconds\n'.format(time.time() - start)
# print(log)
# outfile.write(log)

# log = 'Initial false positive on train data: {0:.6f}\n'.format(my_dc.get_fpr(X_init, Y_init))
# print(log)
# outfile.write(log)

# log = 'Initial memory (Bloom filter): {0:.2f} bytes\n'.format(my_dc.get_size())
# print(log)
# outfile.write(log)

# log = 'Initial memory (Classifier, compressed): {0:.2f} bytes\n'.format(model_size)
# print(log)
# outfile.write(log)

# log = 'Initial memory (Classifier, uncompressed): {0:.2f} bytes\n'.format(model_size_uncompressed)
# print(log)
# outfile.write(log)

# log = 'Initial total memory: {0:.2f} bytes\n'.format(model_size + my_dc.get_size())
# print(log)
# outfile.write(log)

# start = time.time()

# for x, y in zip(X_insert, Y_insert):
#     if np.allclose(y, np.array([1, 0])):
#         my_dc.insert(x)

# log = 'Average insertion time per 1000 elements: {0:.2f} seconds\n'.format((time.time() - start) * 1000 / len(X_insert))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (inserted data): {0:.6f}\n'.format(my_dc.get_fpr(X_insert, Y_insert))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (test data): {0:.6f}\n'.format(my_dc.get_fpr(X_test, Y_test))
# print(log)
# outfile.write(log)

# log = 'False positive after insertion (entire data): {0:.6f}\n'.format(my_dc.get_fpr(np.concatenate((X, X_test)), np.concatenate((Y, Y_test))))
# print(log)
# outfile.write(log)

# log = 'Memory after insertion (Bloom filter): {0:.2f} bytes\n'.format(my_dc.get_size())
# print(log)
# outfile.write(log)

# log = 'Total memory after insertion: {0:.2f} bytes'.format(model_size + my_dc.get_size())
# print(log)
# outfile.write(log)

# outfile.close()

##################################################################################################
# CA-LBF I experiments
# import time
# sys.path.append('bloom_classifier')
# import bloom_classifier as bc

# model.load('models/cifar10.model')
# my_bc = bc.BloomClassifier(model)

# model_size = sys.getsizeof(model)
# model_size_uncompressed = sys.getsizeof(tflearn.variables.get_all_variables())

# outfile = open('outputs/cifar_calbf1_run3.txt', 'w')

# start = time.time()

# my_bc.initialize(X_init, Y_init, m=300)

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

# # model.fit(
# # X_insert,
# # Y_insert,
# # n_epoch=20,
# # shuffle=True,
# # snapshot_epoch=False,
# # snapshot_step=50,
# # validation_set=(X_test_insert, Y_test_insert),
# # show_metric=True,
# # batch_size=128,
# # run_id="cifar10_calbf1")

# # model.save('models/cifar10_calbf1.model')

# model.load('models/cifar10_calbf1.model')

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
import time
sys.path.append('bloom_classifier')
import bloom_classifier as bc

model.load('models/cifar10.model')
my_bc = bc.BloomClassifier(model)

model_size = sys.getsizeof(model)
model_size_uncompressed = sys.getsizeof(tflearn.variables.get_all_variables())

outfile = open('outputs/cifar_calbf2_run3.txt', 'w')

start = time.time()

my_bc.initialize(X_init, Y_init, m=300)

log = 'BF initialization time: {0:.2f} seconds\n'.format(time.time() - start)
print(log)
outfile.write(log)

log = 'Initial false positive on train data: {0:.6f}\n'.format(my_bc.get_fpr(X_init, Y_init))
print(log)
outfile.write(log)

log = 'Initial memory (Bloom filter): {0:.2f} bytes\n'.format(my_bc.get_size())
print(log)
outfile.write(log)

log = 'Initial memory (Classifier, compressed): {0:.2f} bytes\n'.format(model_size)
print(log)
outfile.write(log)

log = 'Initial memory (Classifier, uncompressed): {0:.2f} bytes\n'.format(model_size_uncompressed)
print(log)
outfile.write(log)

log = 'Initial total memory: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
print(log)
outfile.write(log)
start1 = time.time()


tf.reset_default_graph() # reset the model for calbf 2

model.fit(
X_insert,
Y_insert,
n_epoch=20,
shuffle=True,
snapshot_epoch=False,
snapshot_step=50,
validation_set=(X_test_insert, Y_test_insert),
show_metric=True,
batch_size=128,
run_id="cifar10_calbf1")

model.save('models/cifar10_calbf2.model')

# model.load('models/cifar10_calbf2.model')

start2 = time.time()

my_bc.add_data(X_insert, Y_insert, model)

log = 'Average insertion time per 1000 elements: {0:.2f} seconds\n'.format((time.time() - start2) * 1000/ len(X_insert))
print(log)
outfile.write(log)

log = 'Average insertion time per 1000 elements including training: {0:.2f} seconds\n'.format((time.time() - start1) * 1000/ len(X_insert))
print(log)
outfile.write(log)

log = 'False positive after insertion (inserted data): {0:.6f}\n'.format(my_bc.get_fpr(X_insert, Y_insert))
print(log)
outfile.write(log)

log = 'False positive after insertion (test data): {0:.6f}\n'.format(my_bc.get_fpr(X_test, Y_test))
print(log)
outfile.write(log)

log = 'False positive after insertion (entire data): {0:.6f}\n'.format(my_bc.get_fpr(np.concatenate((X, X_test)), np.concatenate((Y, Y_test))))
print(log)
outfile.write(log)

log = 'Memory after insertion (Bloom filter): {0:.2f} bytes\n'.format(my_bc.get_size())
print(log)
outfile.write(log)

log = 'Total memory after insertion: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
print(log)
outfile.write(log)

outfile.close()
