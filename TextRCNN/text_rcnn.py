import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import copy
class TextRCNN:
    def __init__(self, embedding_size, sequence_length, num_classes, w2v_model, rnn_hidden_size, learning_rate, decay_rate, decay_steps, l2_reg_lambda):
        self.embedding_size = embedding_size
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes])
        self.seq_len = tf.placeholder(tf.int32, [None], name = 'seq_len')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name = 'dropout_keep_prob')
        self.first_stage = tf.placeholder(tf.bool, name = 'first_stage')
        self.embeddings_var = tf.Variable(w2v_model, name = 'Word2Vecs')
        self.rnn_hidden_size = rnn_hidden_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.l2_reg_lambda = l2_reg_lambda
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.accuracy = self.accuracy()
        self.train_op = self.train()


    def inference(self):
        embedded_words = tf.nn.embedding_lookup(self.embeddings_var, self.input_x)
        lstm_fw_cell = rnn.BasicLSTMCell(self.rnn_hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.rnn_hidden_size)
        rnn_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_fw_cell, cell_bw = lstm_bw_cell, inputs = embedded_words, sequence_length = self.seq_len, dtype = tf.float32)
        rnn_outputs = tf.concat(rnn_outputs, 2)
        print('embedded_words.shape:', embedded_words.shape)
        print('rnn_outputs.shape:', rnn_outputs.shape)
        outputs = tf.concat([rnn_outputs, embedded_words], 2)
        print('outputs.shape', self.outputs.shape)

        # MaxPooling + Softmax layer
        output_pooling = tf.reduce_max(outputs, axis = 1)
        print('output_pooling.shape', output_pooling.shape)
        h_drop = tf.nn.dropout(output_pooling,keep_prob = self.dropout_keep_prob)
        w = tf.get_variable('rnn_weights',shape=[self.embedding_size + self.rnn_hidden_size * 2, self.num_classes], initializer = tf.random_normal_initializer(stddev=0.1))
        b = tf.get_variable('rnn_b',shape=[self.num_classes])
        logits = tf.nn.xw_plus_b(h_drop, w, b, name = 'logits_rnn_maxpooling')
        
        logits = self.rnn_maxpooling_stage()
        return logits

    def loss(self):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.logits, labels = self.input_y)
        losses = tf.reduce_sum(losses, axis = 1)
        loss = tf.reduce_mean(losses, name = 'loss')
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]) * self.l2_reg_lambda
        loss = loss + l2_losses
        return loss

    def accuracy(self):
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        return accuracy

    def train(self):
        decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase = True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate = decay_learning_rate, optimizer="Adam")
        return train_op
