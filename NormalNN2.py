# coding:utf-8


'''NormalNN is built in order to compare with GAN
It has the same network structure with discriminator of GAN,but it dosent have generater
'''
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

class Train(object):
    def __init__(self):
        sess = tf.Session()
        self.sess = sess
        self.data_dim = 120
        self.batch_size = 6
        self.lr = 0.00001   #learning rate
        self.EPOCH = 500  # the number of max epoch
        self.num_class = 6
        #num of neurons
        self.dis_netnum = {
            'hidden1': 40,
            'hidden2': 20,

        }
        self.regular_num = 0.0001
        self.build_model()  # initializer
        # load test data
        self.testx = np.load('testc.npy')
        self.testy = np.load('tstlc.npy')
        self.testnum = len(self.testx)

    def build_model(self):
        # build  placeholders
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.data_dim], name='real_data')
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class], name='label')
        d_logits_r= self.discriminator('dis', self.x, reuse=False)
        # d_regular = tf.add_n(tf.get_collection('regularizer', 'dis'), 'loss')  # L2 regular loss
        self.crossloss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label * 0.9, logits=d_logits_r)  # Loss
        d_loss_r = tf.reduce_mean(self.crossloss)
        self.d_loss = d_loss_r  # + d_regular
        all_vars = tf.global_variables()
        for v in all_vars:
            print(v)
        self.opt_d = tf.train.AdamOptimizer(self.lr).minimize(self.d_loss)

        # build for test
        test_logits= self.discriminator('dis', self.x, reuse=True)
        test_logits = tf.nn.softmax(test_logits)
        self.test_logits = test_logits
        self.prediction = tf.equal(tf.argmax(test_logits, axis=1), tf.argmax(self.label, axis=1))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.sess.graph.finalize()

    def train(self):
        print('training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        # traindata set
        trainsetx = np.load('trainc.npy')[0:1333]#use part of data
        trainsety = np.load('trlc.npy')[0:1333]
        total = len(trainsetx)
        #for plotting
        plotloss = []
        plotacc = []
        plotx = []
        for epoch in range(self.EPOCH):
            iters = total // self.batch_size
            for idx in range(iters):
                start_t = time.time()
                batchx, batchl = self.getbatch(trainsetx, trainsety, self.batch_size, idx, total)
                d_opt = [self.opt_d, self.d_loss]
                feed = {self.x: batchx, self.label: batchl}
                # update the Discrimater k times
                _, loss_d = self.sess.run(d_opt, feed_dict=feed)
                plotloss.append(loss_d)
                print ("[%3f][epoch:%2d/%2d][iter:%4d/%4d],loss_d:%5f"%
                       (time.time()-start_t, epoch, self.EPOCH,idx,iters, loss_d)),

            test_acc = self.test()
            plotacc.append(test_acc)
            plotx.append(epoch)
            print('test acc:{}'.format(test_acc) + 'temp:%3f' % (epoch))

        print(max(plotacc))
        plt.plot(plotx, plotacc)
        plt.savefig('acc_nn.jpg')
        plt.plot(plotloss)
        plt.savefig('loss_nn.jpg')

    def discriminator(self, name, inputs, reuse):
        #the structure of this model is the same with discriminator of GAN
        if reuse == False:
            with tf.variable_scope(name, reuse=False):
                #layer 1
                stdev = np.sqrt(2. / (self.data_dim + self.dis_netnum['hidden1']))
                size = (self.data_dim, self.dis_netnum['hidden1'])
                dis1weight = tf.get_variable(name='dis1weight', validate_shape=True,
                                             initializer=np.random.uniform(low=-stdev * np.sqrt(3),
                                                                           high=stdev * np.sqrt(3),
                                                                           size=size).astype('float32'))
                dis1biase = tf.get_variable(name='dis1baise', shape=[self.dis_netnum['hidden1']],
                                            initializer=tf.constant_initializer(0.0))
                layer = tf.nn.relu(tf.add(tf.matmul(inputs, dis1weight), dis1biase))
                tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(self.regular_num)(dis1weight))
                #layer 2
                stdev = np.sqrt(2. / (self.dis_netnum['hidden1'] + self.dis_netnum['hidden2']))
                size = (self.dis_netnum['hidden1'], self.dis_netnum['hidden2'])
                dis2weight = tf.get_variable(name='dis2weight', validate_shape=True,
                                             initializer=np.random.uniform(low=-stdev * np.sqrt(3),
                                                                           high=stdev * np.sqrt(3),
                                                                           size=size).astype('float32'))
                dis2biase = tf.get_variable(name='dis2baise', shape=[self.dis_netnum['hidden2']],
                                            initializer=tf.constant_initializer(0.0))
                layer = tf.nn.relu(tf.add(tf.matmul(layer, dis2weight), dis2biase))
                tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(self.regular_num)(dis2weight))
                #layer 3
                stdev = np.sqrt(2. / (self.dis_netnum['hidden2'] + self.num_class))
                size = (self.dis_netnum['hidden2'], self.num_class)
                dis3weight = tf.get_variable(name='dis3weight', validate_shape=True,
                                             initializer=np.random.uniform(low=-stdev * np.sqrt(3),
                                                                           high=stdev * np.sqrt(3),
                                                                           size=size).astype('float32'))
                dis3biase = tf.get_variable(name='dis3baise', shape=[self.num_class],
                                            initializer=tf.constant_initializer(0.0))
                layer = tf.matmul(layer, dis3weight)
                tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(self.regular_num)(dis3weight))

                output = tf.add(layer, dis3biase)
                return output
        else:
            with tf.variable_scope(name, reuse=True):
                dis1weight = tf.get_variable(name='dis1weight')
                dis1biase = tf.get_variable(name='dis1baise')
                layer = tf.nn.relu(tf.add(tf.matmul(inputs, dis1weight), dis1biase))

                dis2weight = tf.get_variable(name='dis2weight')
                dis2biase = tf.get_variable(name='dis2baise')
                layer = tf.nn.relu(tf.add(tf.matmul(layer, dis2weight), dis2biase))

                dis3weight = tf.get_variable(name='dis3weight')
                dis3biase = tf.get_variable(name='dis3baise')
                layer = tf.matmul(layer, dis3weight)

                output = tf.add(layer, dis3biase)
                return output

    # def get_loss(self, logits, layer_out):
    def test(self):
        count = 0.
        for i in range(self.testnum // self.batch_size):
            xtest, ltest = self.getbatch(self.testx, self.testy, self.batch_size, i, self.testnum)
            prediction = self.sess.run(self.prediction, feed_dict={self.x: xtest, self.label: ltest})
            count += np.sum(prediction)
        return count / self.testnum

    def getbatch(self, datasetx, datasety, batchsiaze, i, total):
        batchx = []
        batchy = []
        if (1 + i) * batchsiaze > total - 1:
            for idx in range(i * batchsiaze, total):
                batchx.append(datasetx[idx])
                batchy.append(datasety[idx])
        else:
            for idx in range(i * batchsiaze, (i + 1) * batchsiaze):
                batchx.append(datasetx[idx])
                batchy.append(datasety[idx])
        return batchx, batchy

    def uniform(stdev, size):
        return np.random.uniform(low=-stdev * np.sqrt(3),
                                 high=stdev * np.sqrt(3),
                                 size=size).astype('float32')

