import math
import os
from itertools import cycle

# import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.slim as slim
import torch

import function as func
import SPDnet as spd
from dataset import DREAMERDataset
# from dataset import DREAMERDataset_transfer

# np.random.seed(1500)
# tf.set_random_seed(1500)
# torch.manual_seed(7)

tf.app.flags.DEFINE_string('gpu', '0', 'set gpu')


FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class dnn():
    def __init__(self):
        self.n_hidden_3 = 50
        self.n_hidden_4 = 20
        self.n_class = 2
        self.batch_size = 59

        self.lr = 0.1
        self.max_iteration = 20
        self.dropout_keep_prob = tf.Variable(0.5, trainable=False)
        self.is_training = tf.Variable(True, trainable=False)

        self.sess = tf.Session()
        with tf.name_scope('input'):
            self.X_s = tf.placeholder(
                tf.float32, shape=[self.batch_size, 14, 14])
            self.X_t = tf.placeholder(
                tf.float32, shape=[self.batch_size, 14, 14])
            self.Y_s = tf.placeholder(tf.int64, shape=[self.batch_size, ])
            self.Y_t = tf.placeholder(tf.int64, shape=[self.batch_size, ])
            self.adlamb = tf.placeholder(tf.float32)
        self.ep = {}

        self.weights = {
            'clf': 1,
            'center': 0.01,
            'da_l2': 5e-4,
            'l2': 0.1
        }
        self.momentum = 0.9
        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.train.exponential_decay(
            self.lr,
            global_step=self.global_step,
            decay_steps=270,
            decay_rate=0.1,
            staircase=False)

        # self.train_dataset = DREAMERDataset_transfer(
        #     train=True, transfer=False, source_subject=sub_s, target_subject=sub_t, calibration_num=4, type=class_type, seed=seed)
        # self.transfer_dataset = DREAMERDataset_transfer(
        #     train=True, transfer=True, source_subject=sub_s, target_subject=sub_t, calibration_num=4, type=class_type, seed=seed)
        # self.test_dataset = DREAMERDataset_transfer(
        #     train=False, source_subject=sub_s, target_subject=sub_t, calibration_num=4, type=class_type, seed=seed)

        self.train_dataset = DREAMERDataset(
            train=True, transfer=False)
        self.transfer_dataset = DREAMERDataset(
            train=True, transfer=True)
        self.test_dataset = DREAMERDataset(
            train=False)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.transfer_loader = torch.utils.data.DataLoader(
            dataset=self.transfer_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.train_data = self.train_dataset.train_data
        self.train_labels = self.train_dataset.train_labels

        self.transfer_data = self.transfer_dataset.train_data
        self.transfer_labels = self.transfer_dataset.train_labels

        self.test_data = self.test_dataset.test_data
        self.test_labels = self.test_dataset.test_labels


        self.model()
        self.loss()

    def init_net(self):
        sess = self.sess
        init1 = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()
        sess.run(init1)
        sess.run(init2)
        return

    def adaptation_factor(self, x):
        if x >= 1.0:
            return 1.0
        den = 1.0+math.exp(-10*x)
        lamb = 2.0/den-1.0
        return lamb

    def model(self):
        set_dropout = tf.assign(self.dropout_keep_prob, 0.5)
        clear_dropout = tf.assign(self.dropout_keep_prob, 1.)
        self.ep['set_dropout'] = set_dropout
        self.ep['clear_dropout'] = clear_dropout

        set_training = tf.assign(self.is_training, True)
        clear_training = tf.assign(self.is_training, False)
        self.ep['set_training'] = set_training
        self.ep['clear_training'] = clear_training

        with tf.variable_scope('encoder'):
            X = tf.concat([self.X_s, self.X_t], 0)
            shape = X.get_shape().as_list()
            # BiRe-1
            weight1, weight2 = spd._variable_with_orth_weight_decay(
                'orth_weight0', shape, 14, 10)
            local1 = tf.matmul(tf.matmul(weight2, X),
                               weight1, name='matmulout0')
            local2 = spd._cal_rect_cov(local1)
            # BiRe-2
            shape = local2.get_shape().as_list()
            weight3, weight4 = spd._variable_with_orth_weight_decay(
                'orth_weight1', shape, 10, 5)
            local3 = tf.matmul(tf.matmul(weight4, local2),
                               weight3, name='matmulout1')
            local4 = spd._cal_rect_cov(local3)
            # LogEig Layer
            local5 = spd._cal_log_cov(local4)

            shape = local5.get_shape().as_list()
            feature = tf.reshape(local5, [-1, shape[1]*shape[2]])

        self.ep['feature'] = feature

        with tf.variable_scope('classifier'):
            clf_logits1 = slim.fully_connected(feature,
                                               self.n_hidden_3,
                                               activation_fn=None,
                                               normalizer_fn=None,
                                               scope='clf_h1')
            logits_s, logits_t = tf.split(clf_logits1, 2, 0)
            clf_prob = tf.nn.softmax(clf_logits1, name='clf_prob')

        self.ep['logits'] = clf_logits1
        self.ep['prob'] = clf_prob
        self.ep['logits_s'] = logits_s
        self.ep['logits_t'] = logits_t

        # self.ep['prob'] = prob

        with tf.variable_scope('domain_classifier'):
            clf_logits2 = slim.fully_connected(clf_logits1,
                                               self.n_hidden_4,
                                               activation_fn=tf.nn.relu,
                                               normalizer_fn=None,
                                               scope='da_h1')
            domain_drop = slim.dropout(clf_logits2,
                                       keep_prob=self.dropout_keep_prob,
                                       noise_shape=None,
                                       is_training=self.is_training,
                                       outputs_collections=None,
                                       scope='dropout')
            domain_logits = slim.fully_connected(domain_drop,
                                                 1,
                                                 activation_fn=None,
                                                 normalizer_fn=None,
                                                 scope='da_h2')
            domain_prob = tf.nn.sigmoid(domain_logits)
            domain_logits_s, domain_logits_t = tf.split(domain_logits, 2, 0)

        self.ep['domain_logits_s'] = domain_logits_s
        self.ep['domain_logits_t'] = domain_logits_t

        with tf.variable_scope('centers'):
            centers_s = tf.get_variable('centers1', [self.n_class, self.n_hidden_3], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0))
            centers_t = tf.get_variable('centers2', [self.n_class, self.n_hidden_3], dtype=tf.float32,
                                            initializer=tf.constant_initializer(0))
        self.ep['centers_s'] = centers_s
        self.ep['centers_t'] = centers_t

    def loss(self):
        print ('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        with tf.name_scope('L2_loss'):
            var_list = [v for v in tf.trainable_variables(
            ) if 'domain_classifier' not in v.name]
            L2_list = [v for v in var_list if 'weights' in v.name]
            print ('=================L2_weights=====================')
            print (L2_list)
            L2_loss = tf.add_n(
                [tf.nn.l2_loss(v) for v in L2_list], name='L2_loss')
            self.ep['L2_loss'] = L2_loss

        with tf.name_scope('ad_loss'):
            domain_logits_s = self.ep['domain_logits_s']
            domain_logits_t = self.ep['domain_logits_t']
            D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=domain_logits_t, labels=tf.ones_like(domain_logits_t)))
            D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=domain_logits_s, labels=tf.zeros_like(domain_logits_s)))
            D_loss = 0.1*(D_real_loss+D_fake_loss)
            G_loss = -D_loss
            self.ep['D_loss'] = D_loss
            self.ep['G_loss'] = G_loss

            var_list_da = [v for v in tf.trainable_variables()
                           if 'da' in v.name]
            D_weights = [v for v in var_list_da if 'weights' in v.name]
            D_biases = [v for v in var_list_da if 'biases' in v.name]
            print ('=================Discriminator_weights=====================')
            print (D_weights)
            print ('=================Discriminator_biases=====================')
            print (D_biases)
            Dregloss = self.weights['da_l2'] * tf.reduce_mean(
                [tf.nn.l2_loss(v) for v in var_list_da if 'weights' in v.name])

        D_op1 = tf.train.AdamOptimizer(self.learning_rate, self.momentum).minimize(
            D_loss+Dregloss, var_list=D_weights, global_step=self.global_step)
        D_op2 = tf.train.AdamOptimizer(self.learning_rate*2.0, self.momentum).minimize(
            D_loss+Dregloss, var_list=D_biases, global_step=self.global_step)
        D_op = tf.group(D_op1, D_op2)

        with tf.name_scope('center_loss'):
            dnn_FC_s = self.ep['logits_s']
            dnn_FC_t = self.ep['logits_t']
            centers_s = self.ep['centers_s']
            centers_t = self.ep['centers_t']
            clf_loss_s = func.dce_loss(dnn_FC_s, self.Y_s, centers_s, 1.0)
            clf_loss_t = func.dce_loss(dnn_FC_t, self.Y_t, centers_t, 1.0)
            center_loss = tf.nn.l2_loss(centers_s - centers_t)
            # pl_loss = func.pl_loss(dnn_FC, self.Y, centers)
            self.ep['clf_loss_s'] = clf_loss_s
            self.ep['clf_loss_t'] = clf_loss_t
            self.ep['center_loss'] = center_loss

            train_vars = [v for v in tf.trainable_variables(
            ) if 'domain_classifier' not in v.name]
            print ('=================update ops=====================')
            print (train_vars)

            loss = self.weights['clf'] * (clf_loss_s + clf_loss_t) + self.weights['center'] * center_loss + self.adlamb * G_loss +\
                self.weights['l2'] * L2_loss
            self.ep['loss'] = loss

        train_op = tf.train.AdamOptimizer(self.learning_rate, self.momentum).minimize(loss, var_list=train_vars,
                                                                                      global_step=self.global_step)

        optimizer = tf.group(train_op, D_op)
        self.ep['train_step'] = optimizer

        with tf.name_scope('accuracy'):
            eval_correct1 = func.evaluation(dnn_FC_s, self.Y_s, centers_s)
            predict1, score1 = func.predict_score(dnn_FC_s, centers_s)
            self.ep['src_score'] = score1
            self.ep['src_predict'] = predict1
            self.ep['src_accuracy'] = eval_correct1

            eval_correct2 = func.evaluation(dnn_FC_t, self.Y_t, centers_t)
            predict2, score2 = func.predict_score(dnn_FC_t, centers_t)
            self.ep['tar_score'] = score2
            self.ep['tar_predict'] = predict2
            self.ep['tar_accuracy'] = eval_correct2

            mask1 = tf.cast(tf.less_equal(score1, score2), tf.int32)
            mask2 = tf.cast(tf.greater(score1, score2), tf.int32)
            vote_predict = tf.multiply(
                mask1, predict1)+tf.multiply(mask2, predict2)
            correct = tf.equal(tf.cast(vote_predict, tf.int64),
                               self.Y_s, name='correct')
            all_correct = tf.reduce_mean(
                tf.cast(correct, tf.float32), name='evaluation')
            self.ep['vote_predict'] = vote_predict
            self.ep['vote_accuracy'] = all_correct

    def train(self, iters):
        sess = self.sess
        sess.run(self.ep['set_dropout'])
        sess.run(self.ep['set_training'])
        correct_s = 0
        correct_t = 0
        for step, ((data_s, labels_s), (data_t, labels_t)) in enumerate(zip(self.train_loader, cycle(self.transfer_loader))):
            adlamb = self.adaptation_factor(iters*1.0/self.max_iteration)
            _, L2_loss, clf_loss_s, clf_loss_t, center_loss, D_loss, G_loss = sess.run([self.ep['train_step'], self.ep['L2_loss'], self.ep['clf_loss_s'], self.ep['clf_loss_t'], self.ep['center_loss'], self.ep['D_loss'], self.ep['G_loss']], feed_dict={
                self.X_s: data_s, self.Y_s: labels_s, self.X_t: data_t, self.Y_t: labels_t, self.adlamb: adlamb})
            acc_s, acc_t = sess.run([self.ep['src_accuracy'], self.ep['tar_accuracy']], feed_dict={
                self.X_s: data_s, self.Y_s: labels_s, self.X_t: data_t, self.Y_t: labels_t, self.adlamb: adlamb})
            correct_s += int(acc_s * len(data_s))
            correct_t += int(acc_t * len(data_t))
        acc_s = 100. * correct_s / len(self.train_loader.dataset)
        acc_t = 100. * correct_t / len(self.train_loader.dataset)
        print (acc_s, acc_t)
        # print 'L2_loss', L2_loss
        # print 'clf_loss_s', clf_loss_s
        # print 'clf_loss_t', clf_loss_t
        # print 'center_loss', center_loss
        # print 'D_loss', D_loss
        # print 'G_loss', G_loss

    def test(self):
        sess = self.sess
        sess.run(self.ep['clear_dropout'])
        sess.run(self.ep['clear_training'])
        correct = 0
        predict_loop = []
        for step, (data, labels) in enumerate(self.test_loader):
            vote_acc, vote_predict = sess.run([self.ep['tar_accuracy'], self.ep['vote_predict']], feed_dict={
                self.X_s: data, self.Y_s: labels, self.X_t: data, self.Y_t: labels, self.adlamb: 0})
            correct += int(vote_acc * len(data))
            predict_loop.append(vote_predict)
        acc = 100. * correct / len(self.test_loader.dataset)
        print ('from center compute the acc:', acc)
        y_pred = np.array(predict_loop).flatten()

        return acc, y_pred


def main():
    net = dnn()
    net.init_net()
    last_acc = 0
    last_predict = []

    peak_acc = 0
    peak_predict = []

    for iters in range(net.max_iteration):
        net.train(iters)
        acc,  predict = net.test()
        if acc > peak_acc:
            peak_acc = acc
            peak_predict = predict
    last_acc = acc
    last_predict = predict
    print ('The last acc is :', last_acc)
    print ('The peak acc is :', peak_acc)



if __name__ == '__main__':
    main()
