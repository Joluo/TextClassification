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

class MemNet():
    def __init__(self, doc_len, sent_len, embedding_size, rnn_hidden_size, learning_rate, decay_steps, decay_rate, l2_reg_lambda, w2v_model, mem_size, mem_model, num_classes, fc_layer_size, title_len):
        self.input_title  = tf.placeholder(tf.int32, [None, title_len], name = 'input_title')
        self.input_x = tf.placeholder(tf.int32, [None, doc_len, sent_len], name = 'input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = 'input_y')
        self.mask = tf.placeholder(tf.float32, [None, doc_len, sent_len], name = 'document_Mask')
        self.title_mask = tf.placeholder(tf.float32, [None, title_len], name = 'title_mask')
        self.l1_dropout_keep_prob = tf.placeholder(tf.float32, name = 'l1_dropout_keep_prob')
        self.l2_dropout_keep_prob = tf.placeholder(tf.float32, name = 'l2_dropout_keep_prob')
        print('w2v_model', w2v_model.shape)
        self.w2v = tf.Variable(w2v_model, dtype=tf.float32, name = 'Word2Vecs') #, trainable = False)
        self.mem_size = mem_size
        print('mem', mem_model.shape)
        self.mem = tf.Variable(mem_model, dtype=tf.float32, name = 'memory') #, trainable = False)
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
        with tf.name_scope('doc_han'):
            with tf.name_scope('embedding'):
                xnew = tf.reshape(self.input_x, [-1, self.sent_len])
                print('xnew', xnew.shape)
                we = tf.nn.embedding_lookup(self.w2v, xnew)
                print('we', we.shape)
                we = tf.reshape(we, [-1, self.doc_len, self.sent_len, self.embedding_size])
                print('we', we.shape)
                title_we = tf.nn.embedding_lookup(self.w2v, self.input_title)
                print('title_we', title_we.shape)
            xnew = tf.reshape(we,[-1, self.sent_len, self.embedding_size])
            print('xnew', xnew.shape)
            masknew = tf.reshape(self.mask, [-1, self.sent_len])
            print('mask', self.mask.shape)
            print('masknew', masknew.shape)
            #xnew = tf.unstack(xnew, axis = 1)
            print('xnew', tf.shape(xnew))
  
            self.A0 = Attention(xnew, masknew, scope='A0')

            sentence_emb = tf.reduce_sum(xnew*tf.expand_dims(self.A0.Attn,-1) , reduction_indices=1)
            print('sentence_emb', sentence_emb.shape)
            sentence_emb = tf.reshape(sentence_emb, [-1, self.doc_len, self.embedding_size])
            print('sentence_emb', sentence_emb.shape)
            masknew = tf.cast(tf.reduce_sum(self.mask, reduction_indices= -1)>0,tf.float32)
            print('masknew', masknew.shape)

            self.A1 = Attention(sentence_emb, masknew, scope='A1')        
            output = tf.reduce_sum(sentence_emb*tf.expand_dims(self.A1.Attn,-1) , reduction_indices=1)
            print('output', output.shape)

            self.tA = Attention(title_we, self.title_mask, scope='TA')
            title_emb = tf.reduce_sum(title_we*tf.expand_dims(self.tA.Attn,-1) , reduction_indices=1) 
            print('title_emb', title_emb)

            output = tf.concat([output, title_emb], 1)
            print('output', output.shape)
        with tf.name_scope('mem'):
            output_weights = tf.get_variable(
                    'output_weights',
                    shape=[self.embedding_size * 2, self.fc_layer_size],
                    dtype = tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, dtype = tf.float32, shape=[self.fc_layer_size]), name='output_b')
            doc_output = tf.nn.xw_plus_b(output, output_weights, output_b, name = 'doc_output')
            print('doc_output', doc_output.shape)
            doc_output = tf.nn.dropout(doc_output, self.l1_dropout_keep_prob)

            mem_weights = tf.get_variable(
                    'mem_weights',
                    shape=[self.mem_size, self.fc_layer_size],
                    dtype = tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())
            mem_b = tf.Variable(tf.constant(0.1, dtype = tf.float32, shape=[self.fc_layer_size]), name='mem_b')
            mem_output = tf.nn.xw_plus_b(self.mem, mem_weights, mem_b, name = 'mem_output')
            print('mem_output', mem_output.shape)
            mem_output = tf.nn.dropout(mem_output, self.l2_dropout_keep_prob)
        
            logits = tf.matmul(a=doc_output, b=mem_output, transpose_b = True, name='logits')
            print('logits', logits.shape)
        return logits

    def loss(self):
        with tf.name_scope('loss'):
            log_softmax_output = tf.log(tf.nn.softmax(self.logits)+1e-13)
            loss = - self.num_classes * tf.reduce_mean(tf.multiply(log_softmax_output, self.input_y))
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]) * self.l2_reg_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        with tf.name_scope('train'):
            self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
            train_op = tf.contrib.layers.optimize_loss(self.loss_val, self.global_step, learning_rate=self.decay_learning_rate, optimizer='Adam')
            return train_op

    def accuracy(self):
        with tf.name_scope('accuracy'):
            acc  = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.logits, 1) ,tf.argmax(self.input_y, 1)), tf.float32))
        return acc
