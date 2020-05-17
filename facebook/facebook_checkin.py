import copy
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.append("bloom_classifier")
sys.path.append("dpbf_classifier")

import bloom_classifier as bc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore")  # shouldn't do this

########################################################################################
# Load and prepare data
ds = pd.read_pickle("facebook_checkin.pkl")
X = ds.drop(["place_id"], axis=1)
X = (X - X.mean()) / X.std()
X = X.to_numpy()
Y = ds["place_id"].to_numpy()


X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.25)

X_init, X_insert, Y_init, Y_insert = train_test_split(X, Y, test_size=0.5)
X_test_init, X_test_insert, Y_test_init, Y_test_insert = train_test_split(
    X_test, Y_test, test_size=0.5
)

##################################################################################################
# Build classifier
# model = MLPClassifier(
#     hidden_layer_sizes=(10,),
#     activation='logistic',
#     solver='sgd',
#     warm_start=True) # CA-LBF I

model = MLPClassifier(
    hidden_layer_sizes=(10,), activation="logistic", solver="sgd", warm_start=False
)  # CA-LBF II

##################################################################################################
# Train using classifier. Done once

model.fit(X_init, Y_init)
# pickle.dump(clf, open('models/fbcheckin.model', 'wb'))
# exit()

# ##################################################################################################
# IA-LBF experiments

# import time
# sys.path.append('dpbf_classifier')
# import dpbf_classification as dc

# # model = pickle.load(open('models/fbcheckin.model', 'rb'))
# my_dc = dc.dpbf_logistic(model)

# # print(model.score(X_init_test, Y_init_test))

# model_size = sys.getsizeof(model)

# outfile = open('outputs/fb_ialbf_run1.txt', 'w')

# start = time.time()

# my_dc.initialize(X_init, Y_init, n=10000)

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

# CA-LBF I/II experiments
# check model def

# import time
# sys.path.append('bloom_classifier')
# import bloom_classifier as bc

# my_bc = bc.BloomClassifier(model)

# model_size = sys.getsizeof(model)

# # outfile = open('outputs/fb_calbf1_run1.txt', 'w')
# outfile = open('outputs/fb_calbf2_run3.txt', 'w')

# start = time.time()

# my_bc.initialize(X_init, Y_init, m=250)

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

# log = 'Initial total memory: {0:.2f} bytes\n'.format(model_size + my_bc.get_size())
# print(log)
# outfile.write(log)
# start1 = time.time()

# model.fit(X_insert, Y_insert)

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

# Baseline

sys.path.append("bloom_classifier")

my_bc = bc.BloomClassifier(model)

model_size = sys.getsizeof(model)

outfile = open("outputs/fb_base_run1.txt", "w")

start = time.time()

my_bc.initialize(X_init, Y_init, m=250)

log = "BF initialization time: {0:.2f} seconds\n".format(time.time() - start)
print(log)
outfile.write(log)

log = "Initial false positive on train data: {0:.6f}\n".format(
    my_bc.get_fpr(X_init, Y_init)
)
print(log)
outfile.write(log)

log = "Initial memory (Bloom filter): {0:.2f} bytes\n".format(my_bc.get_size())
print(log)
outfile.write(log)

log = "Initial memory (Classifier, compressed): {0:.2f} bytes\n".format(model_size)
print(log)
outfile.write(log)

log = "Initial total memory: {0:.2f} bytes\n".format(model_size + my_bc.get_size())
print(log)
outfile.write(log)
start1 = time.time()

start2 = time.time()

my_bc.add_data(X_insert, Y_insert, model)

log = "Average insertion time per 1000 elements: {0:.2f} seconds\n".format(
    (time.time() - start2) * 1000 / len(X_insert)
)
print(log)
outfile.write(log)

log = "Average insertion time per 1000 elements including training: {0:.2f} seconds\n".format(
    (time.time() - start1) * 1000 / len(X_insert)
)
print(log)
outfile.write(log)

log = "False positive after insertion (inserted data): {0:.6f}\n".format(
    my_bc.get_fpr(X_insert, Y_insert)
)
print(log)
outfile.write(log)

log = "False positive after insertion (test data): {0:.6f}\n".format(
    my_bc.get_fpr(X_test, Y_test)
)
print(log)
outfile.write(log)

log = "False positive after insertion (entire data): {0:.6f}\n".format(
    my_bc.get_fpr(np.concatenate((X, X_test)), np.concatenate((Y, Y_test)))
)
print(log)
outfile.write(log)

log = "Memory after insertion (Bloom filter): {0:.2f} bytes\n".format(my_bc.get_size())
print(log)
outfile.write(log)

log = "Total memory after insertion: {0:.2f} bytes\n".format(
    model_size + my_bc.get_size()
)
print(log)
outfile.write(log)

outfile.close()

print("\007")
