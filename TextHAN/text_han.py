import tensorflow as tf
import numpy as np

class Attention():
    '''
    Ref:https://github.com/gowthamrang/HAN/blob/master/train.py
    '''
    def __init__(self, input, mask, scope ='A0'):
        assert input.get_shape().as_list()[:-1] == mask.get_shape().as_list() and len(mask.get_shape().as_list()) == 2
        _, steps, embed_dim = input.get_shape().as_list()
        print(steps, embed_dim)
        #trainable variales
        self.u_w = tf.Variable(tf.truncated_normal([1, embed_dim], stddev=0.1),  name='%s_query' %scope, dtype=tf.float32)
        weights = tf.Variable(tf.truncated_normal([embed_dim, embed_dim], stddev=0.1),  name='%s_Weight' %scope, dtype=tf.float32)
        bias = tf.Variable(tf.truncated_normal([1, embed_dim], stddev=0.1),  name='%s_bias' %scope, dtype=tf.float32)
        #equations
        u_i = tf.tanh(tf.matmul(tf.reshape(input,[-1,embed_dim]), weights) + bias)
        u_i = tf.reshape(u_i, [-1,steps, embed_dim])
        distances = tf.reduce_sum(tf.multiply(u_i, self.u_w), reduction_indices=-1)
        self.debug = distances
        self.distances = distances -tf.expand_dims(tf.reduce_max(distances),-1) #avoid exp overflow
        
        expdistance = tf.multiply(tf.exp(self.distances), mask) #
        Denom = tf.expand_dims(tf.reduce_sum(expdistance, reduction_indices=1), 1) + 1e-13 #avoid 0/0 error
        self.Attn = expdistance/Denom
        print('Attn', self.Attn.get_shape())
        return

class TextHAN():
    def __init__(self, doc_len, sent_len, embedding_size, rnn_hidden_size, learning_rate, decay_steps, decay_rate, l2_reg_lambda, w2v_model, num_classes, fc_layer_size):
        self.input_x = tf.placeholder(tf.int32, [None, doc_len, sent_len], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = 'input_y')
        self.mask = tf.placeholder(tf.float32, [None, doc_len, sent_len], name = 'document_Mask')
        self.l1_dropout_keep_prob = tf.placeholder(tf.float32, name = 'l1_dropout_keep_prob')
        self.l2_dropout_keep_prob = tf.placeholder(tf.float32, name = 'l2_dropout_keep_prob')
        self.w2v = tf.Variable(w2v_model, name = 'Word2Vecs')
        self.num_classes = num_classes
        self.global_step = tf.Variable(0, trainable=False)
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.fc_layer_size = fc_layer_size
        self.l2_reg_lambda = l2_reg_lambda
        self.rnn_hidden_size = rnn_hidden_size
        self.sent_len = sent_len
        self.doc_len = doc_len
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.accuracy = self.accuracy()


    def inference(self):
        with tf.name_scope('embedding'):
            xnew = tf.reshape(self.input_x, [-1, self.sent_len])
            print('xnew', xnew.shape)
            we = tf.nn.embedding_lookup(self.w2v, xnew)
            print('we', we.shape)
            we = tf.reshape(we, [-1, self.doc_len, self.sent_len, self.embedding_size])
            print('we', we.shape)
        xnew = tf.reshape(we,[-1, self.sent_len, self.embedding_size])
        print('xnew', xnew.shape)
        masknew = tf.reshape(self.mask, [-1, self.sent_len])
        print('mask', self.mask.shape)
        print('masknew', masknew.shape)
        #xnew = tf.unstack(xnew, axis = 1)
        print('xnew', tf.shape(xnew))
  
        #xnew = tf.nn.dropout(xnew, self.l1_dropout_keep_prob)
        cell_fw = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell = cell_fw,
                                                input_keep_prob = self.l1_dropout_keep_prob)
        cell_bw = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell = cell_bw,
                                                input_keep_prob = self.l1_dropout_keep_prob)
        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, xnew, dtype=tf.float32, scope='L0')
        #print('output', output.shape)

        #output = tf.stack(output, axis=1)
        output =tf.concat(output, 2)
        print('output', output.shape)
        self.A0 = Attention(output, masknew, scope='A0')
        #print('self.A0', tf.shape(self.A0))

        sentence_emb = tf.reduce_sum(output*tf.expand_dims(self.A0.Attn,-1) , reduction_indices=1)
        print('sentence_emb', sentence_emb.shape)
        sentence_emb = tf.reshape(sentence_emb, [-1, self.doc_len, 2 * self.rnn_hidden_size])
        print('sentence_emb', sentence_emb.shape)
        masknew = tf.cast(tf.reduce_sum(self.mask, reduction_indices= -1)>0,tf.float32)
        print('masknew', masknew.shape)
        cell_fw = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell = cell_fw,
                                                input_keep_prob = self.l1_dropout_keep_prob)
        cell_bw = tf.contrib.rnn.GRUCell(self.rnn_hidden_size)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell = cell_bw,
                                                input_keep_prob = self.l1_dropout_keep_prob)

        output, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sentence_emb, dtype = tf.float32, scope='L1')
        #print('output', output.shape)
        #output = tf.stack(output, axis=1)
        output =tf.concat(output, 2)
        print('output', output.shape)

        self.A1 = Attention(output, masknew, scope='A1')        
        #print('self.A1', tf.shape(self.A1))
        output = tf.reduce_sum(sentence_emb*tf.expand_dims(self.A1.Attn,-1) , reduction_indices=1)
        print('output', output.shape)
        
        with tf.variable_scope('fc_layer'):
            # TODO:L2
            '''
            output = tf.nn.dropout(output, self.l1_dropout_keep_prob)
            w = tf.get_variable(
                    'weights_l1',
                    shape=[2 * self.rnn_hidden_size, self.fc_layer_size],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.fc_layer_size]), name='f_b_l1')
            output = tf.nn.xw_plus_b(output, w, b, name = 'fc_layer1')
            output = tf.tanh(output, name = 'fc_layer1')
            print('output', output.shape)
            '''

            output = tf.nn.dropout(output, self.l2_dropout_keep_prob)
            w = tf.get_variable(
                    'weights',
                    shape=[2 * self.rnn_hidden_size, self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='f_b_l2')
            logits = tf.nn.xw_plus_b(output, w, b, name = 'logits')
            print('logits', logits.shape)
            #output = tf.contrib.layers.fully_connected(output, 256, activation_fn = tf.tanh, weights_regularizer = tf.contrib.layers.l2_regularizer(scale = self.l2_reg_lambda))
            #print('output', output.shape)
            #output = tf.nn.dropout(output, self.l2_dropout_keep_prob)
            #logits = tf.contrib.layers.fully_connected(output, self.num_classes, weights_regularizer = tf.contrib.layers.l2_regularizer(scale = self.l2_reg_lambda), activation_fn = None)
            #print('logits', logits.shape)
        return logits

    def loss(self):
        log_softmax_output = tf.log(tf.nn.softmax(self.logits)+1e-13)
        loss = - self.num_classes * tf.reduce_mean(tf.multiply(log_softmax_output, self.input_y))
        #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'fc_net')
        #for w in reg_losses:
        #    shp = w.get_shape().as_list()
        #    print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))
        #reg_loss = tf.reduce_sum(reg_losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]) * self.l2_reg_lambda
        loss = loss + l2_losses
        return loss

    def train(self):
        self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, self.global_step, learning_rate=self.decay_learning_rate, optimizer='Adam')
        return train_op

    def accuracy(self):
        acc  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1) ,tf.argmax(self.input_y, 1)), tf.float32))
        return acc
