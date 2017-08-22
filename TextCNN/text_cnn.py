import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self, sample_len, num_classes, learning_rate, decay_steps, decay_rate,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda, w2v_model):
        self.input_x = tf.placeholder(tf.int32, [None, sample_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
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


    def inference(self):
        with tf.name_scope("embedding"):
            embedded_words = tf.nn.embedding_lookup(self.w2v, self.input_x)
            embedded_words = tf.expand_dims(embedded_words, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                #filter W
                W = tf.get_variable('filter-%s' % filter_size, shape = filter_shape, initializer = tf.random_normal_initializer(stddev=0.1))
                b = tf.get_variable('b-%s' % filter_size, shape=[self.num_filters], initializer = tf.constant_initializer(0))
                conv = tf.nn.conv2d(
                    embedded_words,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Relu
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sample_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
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

