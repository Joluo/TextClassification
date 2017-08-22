import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

'''
@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                               grad,
                                               op.outputs[1],
                                               op.get_attr("ksize"),
                                               op.get_attr("strides"),
                                               padding=op.get_attr("padding"))
'''

class TextCNN(object):
    def __init__(self, sample_len, num_classes, learning_rate, decay_steps, decay_rate,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda, w2v_model):
        self.input_x = tf.placeholder(tf.int32, [None, sample_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
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
        self.w2v = tf.Variable(w2v_model, name = 'Word2Vecs')
        # random initialize word vector
        #self.w2v = tf.Variable(tf.random_uniform([len(w2v_model), embedding_size], -1.0, 1.0), name = 'Word2Vecs')
        self.logits = self.inference()
        #self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        self.accuracy = self.accuracy()

    def batch_norm_wrapper(self, x, n_out, phase_train, decay = 0.9):
        '''
        Ref:https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
        '''
        pop_mean = tf.Variable(tf.zeros(n_out), trainable=False)
        pop_var = tf.Variable(tf.ones(n_out), trainable=False)
        beta = tf.Variable(tf.zeros(n_out))
        gamma = tf.Variable(tf.ones(n_out))
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        def train_update():
            train_mean = tf.assign(pop_mean,
                                   pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var,
                                  pop_var * decay + batch_var * (1 - decay))
            return (train_mean, train_var)

        mean, var = tf.cond(phase_train,
                            train_update,
                            lambda: (pop_mean, pop_var))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed


    def inference(self):
        with tf.name_scope("embedding"):
            embedded_words = tf.nn.embedding_lookup(self.w2v, self.input_x)
            embedded_words = tf.expand_dims(embedded_words, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("aconv-1xd-%s" % filter_size):
                filter_shape = [1, self.embedding_size, 1, self.num_filters]
                W = tf.get_variable('filter-1xd-%s' % filter_size, shape = filter_shape, initializer = tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable('b-1xd-%s' % filter_size, shape=[self.num_filters], initializer = tf.constant_initializer(0))
                conv_1xd = tf.nn.conv2d(
                    embedded_words,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="conv")
                #conv_1xd_bn = self.batch_norm_wrapper(conv_1xd, self.num_filters, self.phase_train)
                output_1xd = tf.nn.relu(tf.nn.bias_add(conv_1xd, b), name="relu")
            with tf.name_scope("aconv-kx1-%s" % filter_size):
                filter_shape = [filter_size, 1, self.num_filters, self.num_filters]
                W = tf.get_variable('filter-kx1-%s' % filter_size, shape = filter_shape, initializer = tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable('b-kx1-%s' % filter_size, shape=[self.num_filters], initializer = tf.constant_initializer(0))
                conv_kx1 = tf.nn.conv2d(
                    output_1xd,
                    W,
                    strides=[1, 1, 1, 1],
                    padding='SAME',
                    name="conv_kx1")
                #conv_kx1_bn = self.batch_norm_wrapper(conv_kx1, self.num_filters, self.phase_train)
                output_kx1 = tf.nn.relu(tf.nn.bias_add(conv_kx1, b), name="relu")
                #pooled = tf.nn.top_k(output_kx1, k=3)
                #print('pooled', tf.shape(pooled))
                '''
                pooled, _ = tf.nn.max_pool_with_argmax(
                            output_kx1,
                            ksize=[1, self.sample_len, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            Targmax=3,
                            name='pool')
                '''
                pooled = tf.nn.max_pool(
                            output_kx1,
                            ksize=[1, self.sample_len, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='pool')
                print('pooled', pooled.shape)
                pooled_outputs.append(pooled)
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            self.h_drop=tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)
        with tf.name_scope("output"):
            '''
            self.W = tf.get_variable(
                'weights',
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
            #logits = tf.matmul(self.h_drop, self.W) + self.b 
            logits = tf.nn.xw_plus_b(self.h_drop, self.W, self.b, name='logits')
            '''
            fw = tf.get_variable(
                'f_weights',
                shape=[num_filters_total, 1024],
                initializer=tf.contrib.layers.xavier_initializer())
            fb = tf.Variable(tf.constant(0.1, shape=[1024]), name='f_b')
            f_output = tf.nn.xw_plus_b(self.h_drop, fw, fb, name='f_outputs')
            self.W = tf.get_variable(
                'weights',
                shape=[1024, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='b')
            logits = tf.nn.xw_plus_b(f_output, self.W, self.b, name='logits')
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
        return train_op

