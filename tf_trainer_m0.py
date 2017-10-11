from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import os

# Memory mgmt
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
#parser.add_argument('-l', '--lr_init', required=True, type=float, help='initial learning rate.')
parser.add_argument('-n', '--train_size', required=False, default=8, type=int, help='training size per decay.')
parser.add_argument('-e', '--epochs', required=False, default=3, type=int, help='Number of epochs to train over.')
args = parser.parse_args()

batch_size = 32*2 # per decay class
nb_epoch = args.epochs
#lr_init = args.lr_init
lr_init = 4.e-4
train_size = args.train_size

import dask.array as da
import h5py

#in_dir = 'IMG'
in_dir = '../EB_H2GG_DiPhotonAllMGG80v2/IMG'
decays = [
'H125GGgluonfusion_Pt25_Eta14_13TeV_TuneCUETP8M1_HighLumiPileUpv3_IMG_RH100_n309k',\
'PromptDiPhotonAll_MGG80toInf_Pt25_Eta14_13TeV_TuneCUETP8M1_HighLumiPileUp_IMG_RH100_n261k']
dsets = [h5py.File('%s/%s.hdf5'%(in_dir,decay)) for decay in decays]

def load_X(dset, start, stop):
    X = dset[start:stop]
    X = (X-121.)/42.
    return X.reshape([-1,1]) 

def load_y(dset, start, stop):
    return dset[start:stop].reshape([-1, 1])

# NOTE: Use of Dask arrays with tflearn.FeedDictFlow completely useless
# as everything is read into memory
# TODO: convert data to TFRecord format

train_start, train_stop, train_chunk_size = 0, train_size*1000, batch_size
#train_start, train_stop, train_chunk_size = 0, 4800, batch_size
n_train = train_stop - train_start
assert n_train % train_chunk_size == 0
assert n_train % batch_size == 0
assert train_chunk_size % batch_size == 0
m0_train = da.from_array(np.concatenate([load_X(dset['m0'],train_start,train_stop) for dset in dsets]), chunks=(train_chunk_size, 1))
y_train = da.from_array(np.concatenate([load_y(dset['y'],train_start,train_stop) for dset in dsets]), chunks=(train_chunk_size, 1))
print("initialized train")

val_start, val_stop, val_chunk_size = 160000, 160000+48000, batch_size
n_val = val_stop - val_start
assert n_val % val_chunk_size == 0
assert n_val % batch_size == 0
assert val_start >= train_stop
m0_val = da.from_array(np.concatenate([load_X(dset['m0'],val_start,val_stop) for dset in dsets]), chunks=(val_chunk_size, 1))
y_val = da.from_array(np.concatenate([load_y(dset['y'],val_start,val_stop) for dset in dsets]), chunks=(val_chunk_size, 1))
print("initialized val")

import tflearn
import tensorflow.contrib.slim as slim

# Model variables
y = tf.placeholder(tf.float32, [None, 1])
m = tf.placeholder(tf.float32, [None, 1])
is_training = tf.placeholder(tf.bool)

def fcnet(inputs):
    net = tflearn.fully_connected(inputs, 128, activation='relu')
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 128, activation='relu')
    net = tflearn.fully_connected(net, 1)
    return net

logits = fcnet(m)
y_pred = tf.nn.sigmoid(logits)

with tf.name_scope('Summaries'):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
    accuracy = tflearn.metrics.binary_accuracy_op(logits, y)

train_op = tf.train.AdamOptimizer(learning_rate=lr_init).minimize(loss)

import time

with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())

    # Initialize data flow mgmt
    coord = tf.train.Coordinator()
    dflow_train = tflearn.data_flow.FeedDictFlow(
            feed_dict={y: y_train, m: m0_train},
            #feed_dict={X: X_train, y: y_train},
            coord=coord,
            shuffle=True,
            continuous=True,
            batch_size=batch_size*len(decays))
    dflow_train.start()

    dflow_val = tflearn.data_flow.FeedDictFlow(
            feed_dict={y: y_val, m: m0_val},
            #feed_dict={X: X_val, y: y_val},
            #shuffle=True,
            coord=coord,
            continuous=True,
            batch_size=batch_size*len(decays))
    dflow_val.start()

    print(">> Training <<<<<<<<")

    for epoch in range(nb_epoch):

        # Run training
        tflearn.is_training(True)
        assert len(dflow_train.batches) == n_train//batch_size
        now = time.time()
        for i in range(n_train//batch_size):
            _, loss_, accuracy_, y_, y_pred_ = sess.run([train_op, loss, accuracy, y, y_pred], feed_dict=dflow_train.next())
            if i % 500 == 0:
                print('%d: Train loss:%f, acc:%f'%(epoch+1,loss_, accuracy_))
        now = time.time() - now
        print('%d: Train time:%f'%(epoch+1,now))

        tflearn.is_training(False)
        n_iter_val = n_val//batch_size
        assert len(dflow_val.batches) == n_iter_val 
        y_val_pred_, y_val_, m0_val_, loss_val_, accuracy_val_ = [], [], [], [], []
        now = time.time()
        for i in range(n_iter_val):
            y_pred_, y_, m0_, loss_, accuracy_ = sess.run([y_pred, y, m, loss, accuracy], feed_dict=dflow_val.next())
            y_val_pred_.append(y_pred_)
            y_val_.append(y_)
            m0_val_.append(m0_)
            loss_val_.append(loss_)
            accuracy_val_.append(accuracy_)
        y_val_pred_ = np.concatenate(y_val_pred_)
        y_val_ = np.concatenate(y_val_)
        m0_val_ = np.concatenate(m0_val_)
        m0_val_ = (42.*m0_val_)+121.
        loss_val_ = np.mean(loss_val_)
        accuracy_val_ = np.mean(accuracy_val_)
        print('%d: Val loss:%f, acc:%f'%(epoch+1,loss_val_, accuracy_val_))
        now = time.time() - now
        print('%d: Val time:%f'%(epoch+1,now))

        #print(np.mean(m0_val_), np.std(m0_val_))
        sct = np.histogram2d(np.squeeze(m0_val_[y_val_ == 1]), np.squeeze(y_val_pred_[y_val_ == 1]), bins=6, range=((80,200), (0.,1.)))[0]
        print(np.uint(np.fliplr(sct).T))

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_val_, y_val_pred_)
    roc_auc = auc(fpr, tpr)
    print("\nVAL ROC AUC:",roc_auc)
