import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class ACBILSTM(object):
    def __init__(self, sample_len, num_classes, learning_rate, decay_steps, decay_rate,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda, w2v_model, rnn_hidden_size):
        self.input_x = tf.placeholder(tf.int32, [None, sample_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.rnn_input_dropout_keep_prob = tf.placeholder(tf.float32, name="rnn_input_dropout_keep_prob")
        self.rnn_output_dropout_keep_prob = tf.placeholder(tf.float32, name="rnn_output_dropout_keep_prob")
        self.phase_train = tf.placeholder(tf.bool, name = 'phase_train')
        self.sample_len = sample_len
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.l2_reg_lambda = l2_reg_lambda
        # user pre-trained word2vec model
        self.w2v = tf.Variable(w2v_model, name = 'Word2Vecs') #, trainable = False)
        self.rnn_hidden_size = rnn_hidden_size
        # random initialize word vector
        #self.w2v = tf.Variable(tf.random_uniform([len(w2v_model), embedding_size], -1.0, 1.0), name = 'Word2Vecs')
        self.all_pop_mean = []
        self.all_pop_var = []
        self.all_beta = []
        self.all_gamma = []
        self.all_batch_mean = []
        self.all_batch_var = []
        self.all_train_mean = []
        self.all_train_var = []
        self.all_mean = []
        self.all_var = []
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        self.accuracy = self.accuracy()

    def batch_norm_wrapper(self, x, n_out, phase_train, decay = 0.5):
        '''
        Ref:https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        '''
        pop_mean = tf.Variable(tf.zeros(n_out), trainable=False)
        pop_var = tf.Variable(tf.ones(n_out), trainable=False)
        beta = tf.Variable(tf.zeros(n_out))
        gamma = tf.Variable(tf.ones(n_out))
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        '''
        self.all_pop_mean.append(pop_mean)
        self.all_pop_var.append(pop_var)
        self.all_beta.append(beta)
        self.all_gamma.append(gamma)
        self.all_batch_mean.append(batch_mean)
        self.all_batch_var.append(batch_var)
        '''

        def train_update():
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            #self.all_train_mean.append(train_mean)
            #self.all_train_var.append(train_var)
            return (train_mean, train_var)

        mean, var = tf.cond(phase_train,
                            train_update,
                            lambda: (pop_mean, pop_var))
        #self.all_mean.append(mean)
        #self.all_var.append(var)
        print('mean', mean.shape)
        print('var', var.shape)
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        print('normed', normed.shape)
        return normed

    def batch_norm(self, x, n_out, phase_train):
        '''
        Ref:https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
        '''
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed

    def inference(self):
        with tf.name_scope("embedding"):
            embedded_words = tf.nn.embedding_lookup(self.w2v, self.input_x)
            print('embedded_words', embedded_words.shape)
            embedded_words = tf.expand_dims(embedded_words, -1)
            print('embedded_words', embedded_words.shape)
        acnn_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("aconv-1xd-%s" % filter_size):
                filter_shape = [1, self.embedding_size, 1, self.num_filters]
                W = tf.get_variable('filter-1xd-%s' % filter_size, shape = filter_shape, initializer = tf.random_normal_initializer(stddev=0.1))
                print('filter-1xd-%s' + str(filter_size) + ':', W.shape)
                #b = tf.get_variable('b-1xd-%s' % filter_size, shape=[self.num_filters], initializer = tf.constant_initializer(0))
                #print('b-1xd-%s' + str(filter_size) , b.shape)
                conv_1xd = tf.nn.conv2d(
                    embedded_words,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv")
                print('conv_1xd:', conv_1xd.shape)

                # BN
                conv_1xd_bn = self.batch_norm(conv_1xd, self.num_filters, self.phase_train)
                print('conv_1xd_bn', conv_1xd_bn.shape)
                
                # Relu
                #output_1xd = tf.nn.relu(tf.nn.bias_add(conv_1xd_bn, b), name="relu")
                output_1xd = tf.nn.relu(conv_1xd_bn, name="relu")
                print('output_1xd:', output_1xd.shape)

            with tf.name_scope("aconv-kx1-%s" % filter_size):
                filter_shape = [filter_size, 1, self.num_filters, self.num_filters]
                W = tf.get_variable('filter-kx1-%s' % filter_size, shape = filter_shape, initializer = tf.random_normal_initializer(stddev=0.1))
                print('filter-kx1-%s' % filter_size, W.shape)
                #b = tf.get_variable('b-kx1-%s' % filter_size, shape=[self.num_filters], initializer = tf.constant_initializer(0))
                #print('b-kx1-%s' % filter_size, b.shape)
                conv_kx1 = tf.nn.conv2d(
                    output_1xd,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="conv_kx1")
                print('conv_kx1:', conv_kx1.shape)

                # BN
                conv_kx1_bn = self.batch_norm(conv_kx1, self.num_filters, self.phase_train)
                print('conv_kx1_bn', conv_kx1_bn.shape)

                #output_kx1 = tf.nn.relu(tf.nn.bias_add(conv_kx1_bn, b), name="relu")
                output_kx1 = tf.nn.relu(conv_kx1_bn, name="relu")
                print('output_kx1', output_kx1.shape)
                output_kx1 = tf.squeeze(output_kx1, [2])
                print('output_kx1', output_kx1.shape)
                acnn_outputs.append(output_kx1)
        self.acnn_output = tf.concat(acnn_outputs, 2)
        print('acnn_output', self.acnn_output.shape)

        #w_size = self.acnn_output.shape[1] * self.acnn_output.shape[2]
        #self.acnn_output_flat = tf.reshape(self.acnn_output, tf.stack([-1, w_size]))
        #print('acnn_output_flat', self.acnn_output_flat.shape)

        #self.acnn_output_drop_flat = tf.nn.dropout(self.acnn_output_flat, self.rnn_input_dropout_keep_prob)
        #print('acnn_output_drop_flat', self.acnn_output_drop_flat.shape)

        #self.acnn_output_drop = tf.reshape(self.acnn_output_drop_flat, tf.stack([-1, self.acnn_output.shape[1], self.acnn_output.shape[2]]))
        #print('acnn_output_drop', self.acnn_output_drop.shape)

        with tf.variable_scope('LSTM_0'):
            lstm_fw_cell_0 = rnn.DropoutWrapper(cell = rnn.BasicLSTMCell(self.rnn_hidden_size), 
                                                input_keep_prob = self.rnn_input_dropout_keep_prob, 
                                                output_keep_prob = self.rnn_output_dropout_keep_prob)
            lstm_bw_cell_0 = rnn.DropoutWrapper(cell = rnn.BasicLSTMCell(self.rnn_hidden_size), 
                                                input_keep_prob = self.rnn_input_dropout_keep_prob,
                                                output_keep_prob = self.rnn_output_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell_0, 
                                                                  cell_bw = lstm_bw_cell_0, 
                                                                  inputs = self.acnn_output, 
                                                                  dtype = tf.float32)
            self.rnn_output = tf.concat(self.rnn_outputs, 2)
            print('self.rnn_output', self.rnn_output.shape)

        '''
        with tf.variable_scope('LSTM_1'):
            lstm_fw_cell_1 = rnn.DropoutWrapper(cell = rnn.BasicLSTMCell(self.rnn_hidden_size),
                                                output_keep_prob = self.rnn_output_dropout_keep_prob)
            lstm_bw_cell_1 = rnn.DropoutWrapper(cell = rnn.BasicLSTMCell(self.rnn_hidden_size),
                                                output_keep_prob = self.rnn_output_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell_1, 
                                                                  cell_bw = lstm_bw_cell_1, 
                                                                  inputs = self.rnn_output, 
                                                                  dtype = tf.float32)
            self.rnn_output = tf.concat(self.rnn_outputs, 2)
            print('self.rnn_output', self.rnn_output.shape)

        with tf.variable_scope('LSTM_2'):
            lstm_fw_cell_2 = rnn.DropoutWrapper(cell = rnn.BasicLSTMCell(self.rnn_hidden_size),
                                                output_keep_prob = self.rnn_output_dropout_keep_prob)
            lstm_bw_cell_2 = rnn.DropoutWrapper(cell = rnn.BasicLSTMCell(self.rnn_hidden_size),
                                                output_keep_prob = self.rnn_output_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell_2, 
                                                                  cell_bw = lstm_bw_cell_2, 
                                                                  inputs = self.rnn_output, 
                                                                  dtype = tf.float32)
            self.rnn_output = tf.concat(self.rnn_outputs, 2)
            print('self.rnn_output', self.rnn_output.shape)

        with tf.variable_scope('LSTM_3'):
            lstm_fw_cell_3 = rnn.DropoutWrapper(cell = rnn.BasicLSTMCell(self.rnn_hidden_size),
                                                output_keep_prob = self.rnn_output_dropout_keep_prob)
            lstm_bw_cell_3 = rnn.DropoutWrapper(cell = rnn.BasicLSTMCell(self.rnn_hidden_size),
                                                output_keep_prob = self.rnn_output_dropout_keep_prob)
            self.rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell_3, 
                                                                  cell_bw = lstm_bw_cell_3, 
                                                                  inputs = self.rnn_output, 
                                                                  dtype = tf.float32)
            self.rnn_output = tf.concat(self.rnn_outputs, 2)
            print('self.rnn_output', self.rnn_output.shape)            
        '''

        w_size = self.rnn_output.shape[1] * self.rnn_output.shape[2]
        print('w_size', w_size)

        self.rnn_output_flat = tf.reshape(self.rnn_output, tf.stack([-1, w_size]))
        print('rnn_output_flat', self.rnn_output_flat.shape)

        #self.rnn_output_flat = tf.nn.dropout(self.rnn_output_flat, self.rnn_output_dropout_keep_prob)

        W = tf.get_variable('weights', shape = [self.rnn_output_flat.shape[1], self.num_classes], initializer = tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable('bias', shape=[self.num_classes], initializer = tf.constant_initializer(0))
        logits = tf.nn.xw_plus_b(self.rnn_output_flat, W, b, name = 'logits')
        return logits

    def loss(self):
        with tf.name_scope("loss"):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.input_y, logits = self.logits)
            losses = tf.reduce_sum(losses, axis = 1)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]) * self.l2_reg_lambda
            loss=loss+l2_losses
        return loss

    def accuracy(self):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        return accuracy

    def train(self):
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=self.decay_learning_rate, optimizer="Adam")
        #train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss = self.loss_val, global_step=self.global_step)
        return train_op

