#coding:utf-8
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time
# temp = 0.89
class Train(object):
    def __init__(self):
        sess=tf.Session()
        self.sess = sess
        self.data_dim = 60   # the size of image
        self.trainable = True
        self.batch_size = 8  # must be even number
        self.lr = 0.00001
        self.mm = 0.5      # momentum term for adam
        self.z_dim =5   # the dimension of noise z
        self.EPOCH = 1000  # the number of max epoch
        self.LAMBDA = 0.1  # parameter of WGAN-GP
        self.model = 'DCGAN'  # 'DCGAN' or 'WGAN'
        self.dim = 1       # RGB is different with gray pic
        self.num_class = 4
        self.regular_num = 0.0001
        self.load_model = False
        self.gen_netnum = {
            'hidden1': 20,
            'hidden2': 40,
        }
        self.dis_netnum = {
            'hidden1': 100,
            'hidden2': 40,
        }

        self.build_model()  # initializer
        self.testx = np.load('test.npy')
        self.testy = np.load('tstl.npy')
        self.testnum=len(self.testx)


    def build_model(self):
        # build  placeholders
        self.x=tf.placeholder(tf.float32,shape=[self.batch_size,self.data_dim],name='real_data')
        self.z = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim], name='noise')#noise
        self.label = tf.placeholder(tf.float32, shape=[self.batch_size, self.num_class - 1], name='label')
        self.flag = tf.placeholder(tf.float32, shape=[], name='flag')

        # define the network
        self.fake_data = self.generator('gen', self.z, reuse=False)
        d_logits_r, layer_out_r = self.discriminator('dis', self.x, reuse=False)
        d_logits_f, layer_out_f = self.discriminator('dis', self.fake_data, reuse=True)
        d_regular = tf.add_n(tf.get_collection('regularizer', 'dis'), 'loss')  # L2 regular loss

        # caculate the unsupervised loss
        un_label_r = tf.concat([tf.ones_like(self.label), tf.zeros(shape=(self.batch_size, 1))], axis=1)
        un_label_f = tf.concat([tf.zeros_like(self.label), tf.ones(shape=(self.batch_size, 1))], axis=1)
        logits_r, logits_f = tf.nn.softmax(d_logits_r), tf.nn.softmax(d_logits_f)
        d_loss_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=un_label_r*0.9, logits=d_logits_r))
        d_loss_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=un_label_f*0.9, logits=d_logits_f))

        # feature match
        f_match = tf.constant(0., dtype=tf.float32)
        for i in range(3):
            f_match += tf.reduce_mean(tf.multiply(layer_out_f[i]-layer_out_r[i], layer_out_f[i]-layer_out_r[i]))

        # caculate the supervised loss
        s_label = tf.concat([self.label, tf.zeros(shape=(self.batch_size,1))], axis=1)
        s_l_r = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=s_label*0.9, logits=d_logits_r))
        s_l_f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=un_label_f*0.9, logits=d_logits_f))  # same as d_loss_f
        self.d_l_1, self.d_l_2 = d_loss_r + d_loss_f, s_l_r
        self.d_loss = d_loss_r + d_loss_f + s_l_r*self.flag*10  + d_regular
        self.g_loss = d_loss_f + 0.01*f_match

        all_vars = tf.global_variables()
        g_vars = [v for v in all_vars if 'gen' in v.name]
        d_vars = [v for v in all_vars if 'dis' in v.name]
        for v in all_vars:
            print( v)
        if self.model == 'DCGAN':
            self.opt_d = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(self.lr, beta1=self.mm).minimize(self.g_loss, var_list=g_vars)
        elif self.model == 'WGAN_GP':
            self.opt_d = tf.train.AdamOptimizer(1e-6, beta1=0.5, beta2=0.9).minimize(self.d_loss, var_list=d_vars)
            self.opt_g = tf.train.AdamOptimizer(1e-6, beta1=0.5, beta2=0.9).minimize(self.g_loss, var_list=g_vars)
        else:
            print ('model can only be "DCGAN","WGAN_GP" !')
            return
        # test
        test_logits, _ = self.discriminator('dis', self.x, reuse=True)

        test_logits = tf.nn.softmax(test_logits)

        temp = tf.reshape(test_logits[:, -1],shape=[self.batch_size, 1])
        for i in range(4):
            self.temp = tf.concat([temp, tf.reshape(test_logits[:, -1],shape=[self.batch_size, 1])], axis=1)
        test_logits -= temp
        self.prediction = tf.nn.in_top_k(test_logits, tf.argmax(s_label, axis=1), 1)
        self.saver = tf.train.Saver()
        if not self.load_model:
            init = tf.global_variables_initializer()
            self.sess.run(init)
        elif self.load_model:
            self.saver.restore(self.sess, os.getcwd()+'/model_saved/model.ckpt')
            print('model load done')
        self.sess.graph.finalize()

    def train(self):
        if not os.path.exists('model_saved'):
            os.mkdir('model_saved')
        if not os.path.exists('gen_picture'):
            os.mkdir('gen_picture')
        noise = np.random.normal(-1, 1, [self.batch_size, self.z_dim])
        plotx=[]
        ploty=[]
        plotlossg=[]
        plotlossd=[]
        temp = 0.80
        print ('training')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(self.sess, coord)
        #traindata set
        trainsetx = np.load('train.npy')
        trainsety = np.load('trl.npy')
        total=len(trainsetx)
        for epoch in range(self.EPOCH):
            # iters = int(156191//self.batch_size)
            iters = total//self.batch_size
            for idx in range(iters):
                start_t = time.time()
                flag = 1 if idx < 334 else 0 # use 90 of data for training
                batchx,batchl=self.getbatch(trainsetx,trainsety,self.batch_size,idx,total)
                g_opt = [self.opt_g, self.g_loss]
                d_opt = [self.opt_d, self.d_loss, self.d_l_1, self.d_l_2]
                feed = {self.x:batchx, self.z:noise, self.label:batchl, self.flag:flag}
                # update the Discrimater k times
                _, loss_d, d1,d2 = self.sess.run(d_opt, feed_dict=feed)
                # update the Generator one time
                _, loss_g = self.sess.run(g_opt, feed_dict=feed)
                '''
                print ("[%3f][epoch:%2d/%2d][iter:%4d/%4d],loss_d:%5f,loss_g:%4f, d1:%4f, d2:%4f"%
                       (time.time()-start_t, epoch, self.EPOCH,idx,iters, loss_d, loss_g,d1,d2)), 'flag:',flag
                '''
                plotlossg.append(loss_g)
               # plotlossg.append(loss_d)

            test_acc = self.test()
            print ('test acc:{}'.format(test_acc)+ 'temp:%3f'%(epoch))
            ploty.append(test_acc)
            plotx.append(epoch)
            '''
            if test_acc > temp:
                print ('model saving..............')
                path = os.getcwd() + '/model_saved'
                save_path = os.path.join(path, "model.ckpt")
                self.saver.save(self.sess, save_path=save_path)
                print ('model saved...............')
                temp = test_acc
            '''

        plt.plot(plotx, ploty)
        plt.savefig('accsave.jpg')
        plt.plot(plotlossg)
        plt.savefig('loo_g_save.jpg')
       # plt.plot(plotlossd)
        #plt.savefig('loo_d_save.jpg')
        print(max(ploty))
    # output = conv2d('Z_cona{}'.format(i), output, 3, 64, stride=1, padding='SAME')

    def generator(self,name, noise, reuse):
        #3 layers or 5 layers
        with tf.variable_scope(name,reuse=reuse):
            l = self.batch_size
            gen1weight=tf.get_variable(name='gen1weight', shape=[self.z_dim,self.gen_netnum['hidden1']], initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
            gen1baise=tf.get_variable(name='gen1baise', shape=[self.gen_netnum['hidden1']],initializer=tf.constant_initializer(0.0))
            layer = tf.nn.relu(tf.add(tf.matmul(noise, gen1weight),gen1baise))


            gen2weight=tf.get_variable(name='gen2weight',shape=[self.gen_netnum['hidden1'], self.gen_netnum['hidden2']],
                                                            initializer=tf.random_normal_initializer(mean=0,stddev=1))
            gen2baise=tf.get_variable(name='gen2baise', shape=[self.gen_netnum['hidden2']],
                                                      initializer=tf.constant_initializer(0.0))
            layer = tf.nn.relu(tf.add(tf.matmul(layer, gen2weight),gen2baise))

            gen3weight=tf.get_variable(name='gen3weight',shape=[self.gen_netnum['hidden2'], self.data_dim],
                                                                       initializer=tf.random_normal_initializer(mean=0,stddev=1))
            gen3baise=tf.get_variable(name='gen3baise', shape=[self.data_dim],
                                                      initializer=tf.random_normal_initializer(mean=0, stddev=0.01))

            layer = tf.add(tf.matmul(layer, gen3weight),gen3baise)

            return tf.nn.tanh(layer)

    def discriminator(self, name, inputs, reuse):
        #3 layers or 5 layers
        if reuse==False:
            with tf.variable_scope(name,reuse=False):
                out = []
                stdev = np.sqrt(2. / (self.data_dim + self.dis_netnum['hidden1']))
                size = (self.data_dim, self.dis_netnum['hidden1'])
                dis1weight = tf.get_variable(name='dis1weight', validate_shape=True,
                                             initializer=np.random.uniform(low=-stdev * np.sqrt(3),
                                                                           high=stdev * np.sqrt(3),
                                                                           size=size).astype('float32'))
                dis1biase=tf.get_variable(name='dis1baise', shape=[self.dis_netnum['hidden1']], initializer=tf.constant_initializer(0.0))
                layer = tf.nn.relu(tf.add(tf.matmul(inputs, dis1weight),dis1biase))
                tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(self.regular_num)(dis1weight))
                out.append(layer)


                #layer = tf.nn.dropout(layer, keep_prob=0.8)
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
                #tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(self.regular_num)(dis2biase))
                out.append(layer)
                #layer = tf.nn.dropout(layer, keep_prob=0.8)
                stdev = np.sqrt(2. / (self.dis_netnum['hidden2'] + self.num_class))
                size = (self.dis_netnum['hidden2'], self.num_class)
                dis3weight = tf.get_variable(name='dis3weight', validate_shape=True,
                                             initializer=np.random.uniform(low=-stdev * np.sqrt(3),
                                                                           high=stdev * np.sqrt(3),
                                                                           size=size).astype('float32'))
                dis3biase = tf.get_variable(name='dis3baise', shape=[self.num_class],
                                            initializer=tf.constant_initializer(0.0))
                tf.add_to_collection('regularizer', tf.contrib.layers.l2_regularizer(self.regular_num)(dis3weight))
                layer = tf.add(tf.matmul(layer, dis3weight), dis3biase)
                out.append(layer)
                output =layer
                return output, out
        else:
            with tf.variable_scope(name,reuse=True):
                out = []
                dis1weight=tf.get_variable(name='dis1weight')
                dis1biase=tf.get_variable(name='dis1baise')
                layer = tf.nn.relu(tf.add(tf.matmul(inputs, dis1weight),dis1biase))
                out.append(layer)


                #layer = tf.nn.dropout(layer, keep_prob=0.8)
                dis2weight = tf.get_variable(name='dis2weight')
                dis2biase = tf.get_variable(name='dis2baise')
                layer = tf.nn.relu(tf.add(tf.matmul(layer, dis2weight), dis2biase))
                out.append(layer)
                #layer = tf.nn.dropout(layer, keep_prob=0.8)
                dis3weight = tf.get_variable(name='dis3weight')
                dis3biase = tf.get_variable(name='dis3baise')
                layer = tf.add(tf.matmul(layer, dis3weight), dis3biase)
                out.append(layer)
                output =layer
                return output, out

    # def get_loss(self, logits, layer_out):
    def test(self):
        count = 0.
        for i in range(self.testnum//self.batch_size):
            xtest,ltest=self.getbatch(self.testx,self.testy,self.batch_size,i,self.testnum)
            prediction = self.sess.run(self.prediction, feed_dict={self.x:xtest, self.label:ltest})
            count += np.sum(prediction)
        return count/self.testnum

    def getbatch(self,datasetx,datasety,batchsiaze,i,total):
        batchx=[]
        batchy=[]
        if (1+i)*batchsiaze > total-1:
            for idx in range(i*batchsiaze,total):
                batchx.append(datasetx[idx])
                batchy.append(datasety[idx])
        else:
            for idx in range(i*batchsiaze,(i+1)*batchsiaze):
                batchx.append(datasetx[idx])
                batchy.append(datasety[idx])
        return batchx,batchy

Ganmodel=Train()

Ganmodel.train()

