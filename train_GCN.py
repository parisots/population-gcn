# Copyright (c) 2016 Thomas Kipf
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira
# Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated
# documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.


from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

import random
from gcn.utils import *
from gcn.models import MLP, Deep_GCN
import sklearn.metrics

def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask


def run_training(adj, features, labels, idx_train, idx_val, idx_test,
                 params, sex_data=None, stratify=False, fold_index=None,
                 ):
    # Set random seed
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    tf.set_random_seed(params['seed'])

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model', params['model'],
                        'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
    flags.DEFINE_float('learning_rate', params['lrate'],
                       'Initial learning rate.')
    flags.DEFINE_integer('epochs', params['epochs'],
                         'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', params['hidden'],
                         'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', params['dropout'],
                       'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', params['decay'],
                       'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', params['early_stopping'],
                         'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', params['max_degree'],
                         'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('depth', params['depth'], 'Depth of Deep GCN')

    # Create test, val and train masked variables
    y_train, y_val, y_test, train_mask, val_mask, test_mask = \
        get_train_test_masks(
        labels, idx_train, idx_val, idx_test)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = Deep_GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = Deep_GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for GCN model ')

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in
                    range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32,
                                          shape=tf.constant(features[2],
                                                            dtype=tf.int64)),
        'phase_train': tf.placeholder_with_default(False, shape=()),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
    # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1],
                       depth=FLAGS.depth, logging=True)

    # Initialize session
    saver = tf.train.Saver()
    sess = tf.Session()

    def get_stratified_data(y_train, train_idx, sex_labels):
        female_n = sex_labels[train_idx, 1].sum()
        female_idx = np.intersect1d(train_idx,
                                    np.argwhere(sex_labels[:, 1] == 1))
        male_idx = np.intersect1d(train_idx,
                                  np.argwhere(sex_labels[:, 0] == 1))
        male_idx_stratified = np.random.choice(male_idx, int(female_n))

        idx_stratified = np.concatenate((female_idx, male_idx_stratified))
        return sample_mask(idx_stratified, labels.shape[0])

    # Define model evaluation function
    def evaluate(feats, graph, label, mask, placeholder):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(feats, graph, label, mask,
                                            placeholder)
        feed_dict_val.update({placeholder['phase_train'].name: False})
        outs_val = sess.run([model.loss, model.accuracy, model.predict()],
                            feed_dict=feed_dict_val)

        # Compute the area under curve
        pred = outs_val[2]
        pred = pred[np.squeeze(np.argwhere(mask == 1)), :]
        lab = label
        lab = lab[np.squeeze(np.argwhere(mask == 1)), :]
        auc = sklearn.metrics.roc_auc_score(np.squeeze(lab), np.squeeze(pred))
        return outs_val[2], outs_val[0], outs_val[1], auc, (
                    time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(params['epochs']):

        t = time.time()
        # Construct feed dictionary
        if stratify:
            train_mask_str = get_stratified_data(y_train, idx_train, sex_data)
            feed_dict = construct_feed_dict(features, support, y_train,
                                            train_mask_str, placeholders)
        else:
            feed_dict = construct_feed_dict(features, support, y_train,
                                            train_mask, placeholders)

        feed_dict.update({placeholders['dropout']: FLAGS.dropout,
                          placeholders['phase_train']: True})

        # Training step
        outs = sess.run(
            [model.opt_op, model.loss, model.accuracy, model.predict()],
            feed_dict=feed_dict)
        pred_train = outs[3]
        pred_train = pred_train[np.squeeze(np.argwhere(train_mask == 1)), :]
        labs = y_train
        labs = labs[np.squeeze(np.argwhere(train_mask == 1)), :]
        train_auc = sklearn.metrics.roc_auc_score(np.squeeze(labs),
                                                  np.squeeze(pred_train))

        # Validation
        _, cost, acc, auc, duration = evaluate(features, support, y_val,
                                               val_mask, placeholders)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=",
              "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "train_auc=",
              "{:.5f}".format(train_auc), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "val_auc=",
              "{:.5f}".format(auc), "time=",
              "{:.5f}".format(time.time() - t + duration))

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
                cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    if fold_index != None:  # Saving models' parameters
        saver.save(sess, 'GCN_asd_males_' + str(fold_index),
                   global_step=params['epochs'])  ###

    print("Optimization Finished!")

    # Testing
    sess.run(tf.local_variables_initializer())
    pred, test_cost, test_acc, test_auc, test_duration = evaluate(features,
                                                                  support,
                                                                  y_test,
                                                                  test_mask,
                                                                  placeholders)

    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "auc=", "{:.5f}".format(test_auc))

    return pred, test_acc, test_auc, pred_train
