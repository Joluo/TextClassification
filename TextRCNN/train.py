import tensorflow as tf
from tflearn.data_utils import pad_sequences
import numpy as np
from text_rcnn import TextRCNN
from data_helper import load_w2v, loadSamples
import math
import os
import time
import datetime

FLAGS=tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('word_embedding_file', 'zhihu/data/ieee_zhihu_cup/word_embedding.txt', 'word embedding file')
tf.app.flags.DEFINE_string('training_data', 'zhihu/data/ieee_zhihu_cup/question_train_set.txt', 'total trianing data of title text')
tf.app.flags.DEFINE_string('label_file', 'zhihu/data/ieee_zhihu_cup/question_topic_train_set.txt', 'trianing sample label file')
tf.app.flags.DEFINE_string('label_map', 'zhihu/data/samples/id_map', 'label id mapping')
tf.app.flags.DEFINE_boolean('shuffle', True, 'does shuffle training samples')
tf.app.flags.DEFINE_float('valid_rate', 0.1, 'valid sample rate')

tf.app.flags.DEFINE_integer('embedding_size', 256, 'embedding size of words')
tf.app.flags.DEFINE_integer('sample_len', 150, 'max sample len')
tf.app.flags.DEFINE_integer('rnn_hidden_size', 128, 'max sample len')
tf.app.flags.DEFINE_integer('num_classes', 1999, 'number of classes')
tf.app.flags.DEFINE_integer('batch_size', 512, 'batch size')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
tf.app.flags.DEFINE_integer('decay_steps', 5000, 'decay steps')
tf.app.flags.DEFINE_float('decay_rate', 0.75, 'decay rate')
tf.app.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2 regular para')
tf.app.flags.DEFINE_integer('num_epochs', 40, 'number of epochs')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
tf.app.flags.DEFINE_integer('print_stats_every', 100, 'Print training stats numbers after this many steps')
tf.app.flags.DEFINE_integer('evaluate_every', 400, 'Evaluate model on dev set after this many steps')
tf.app.flags.DEFINE_integer('checkpoint_every', 400, 'Save model after this many steps')
tf.app.flags.DEFINE_integer('num_checkpoints', 30, 'Number of checkpoints to store')
tf.app.flags.DEFINE_boolean('save_best_model', True, 'if just save best perf models on valid set')

def main(_):
    print('Loading word2vec model finished:%s' % (FLAGS.word_embedding_file))
    w2v_model, word2id = load_w2v(FLAGS.word_embedding_file, 256)
    print('Load word2vec model finished')
    print('Loading train/valid samples:%s' % (FLAGS.training_data))
    train_x, train_y, train_len, valid_x, valid_y, valid_len = loadSamples(FLAGS.training_data, FLAGS.label_file, FLAGS.label_map, word2id, FLAGS.valid_rate, FLAGS.num_classes)
    print('Load train/valid samples finished')
    train_x = pad_sequences(train_x, maxlen=FLAGS.sample_len, value = 0.)
    valid_x = pad_sequences(valid_x, maxlen=FLAGS.sample_len, value = 0.)

    train_sample_size = len(train_x)
    dev_sample_size = len(valid_x)
    print('Training sample size:%d' % (train_sample_size))
    print('Valid sample size:%d' % (dev_sample_size))

    timestamp = str(int(time.time()))
    runs_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs'))
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    out_dir = os.path.abspath(os.path.join(runs_dir, timestamp))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(), tf.device('/gpu:1'):
            text_rcnn = TextRCNN(
                embedding_size = FLAGS.embedding_size,
                sequence_length = FLAGS.sample_len,
                num_classes = FLAGS.num_classes,
                w2v_model = w2v_model,
                rnn_hidden_size = FLAGS.rnn_hidden_size,
                learning_rate = FLAGS.learning_rate,
                decay_rate = FLAGS.decay_rate,
                decay_steps = FLAGS.decay_steps,
                l2_reg_lambda = FLAGS.l2_reg_lambda)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            loss_summary = tf.summary.scalar('loss', text_rcnn.loss_val)
            acc_summary = tf.summary.scalar('accuracy', text_rcnn.accuracy)
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            sess.run(tf.global_variables_initializer())
            total_loss = 0.
            total_acc = 0.
            total_step = 0.
            best_valid_acc = 0.
            best_valid_loss = 1000.
            best_valid_zhihu_score = 0.
            this_step_valid_acc = 0.
            this_step_valid_loss = 0.
            this_step_zhihu_score = 0.
            valid_loss_summary = tf.summary.scalar('loss', this_step_valid_loss)
            valid_acc_summary = tf.summary.scalar('accuracy', this_step_valid_acc)
            valid_zhihu_score_summary = tf.summary.scalar('zhihu_score', this_step_zhihu_score)
            valid_summary_op = tf.summary.merge([valid_loss_summary, valid_acc_summary, valid_zhihu_score_summary])
            for epoch in range(0, FLAGS.num_epochs):
                print('epoch:' + str(epoch))
                batch_step = 0
                batch_loss = 0.
                batch_acc = 0.
                for start, end in zip(range(0, train_sample_size, FLAGS.batch_size), range(FLAGS.batch_size, train_sample_size, FLAGS.batch_size)):
                    if total_step > 20000:
                        feed_dict = {
                            text_rcnn.input_x: train_x[start:end],
                            text_rcnn.input_y: train_y[start:end],
                            text_rcnn.seq_len: train_len[start:end],
                            text_rcnn.first_stage: False,
                            text_rcnn.dropout_keep_prob: 0.5
                        }
                    else:
                        feed_dict = {
                            text_rcnn.input_x: train_x[start:end],
                            text_rcnn.input_y: train_y[start:end],
                            text_rcnn.seq_len: train_len[start:end],
                            text_rcnn.first_stage: True,
                            text_rcnn.dropout_keep_prob: 0.5
                        }
                    loss, acc, step, summaries, _ = sess.run([text_rcnn.loss_val, text_rcnn.accuracy, text_rcnn.global_step, train_summary_op, text_rcnn.train_op], feed_dict)
                    train_summary_writer.add_summary(summaries, step)
                    total_loss += loss
                    total_acc += acc
                    batch_loss += loss
                    batch_acc += acc
                    batch_step += 1
                    total_step += 1.
                    if batch_step % FLAGS.print_stats_every == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print('[%s]Epoch:%d\tBatch_Step:%d\tTrain_Loss:%.4f/%.4f/%.4f\tTrain_Accuracy:%.4f/%.4f/%.4f' % (time_str, epoch, batch_step, loss, batch_loss / batch_step, total_loss / total_step, acc, batch_acc / batch_step, total_acc / total_step))
                    this_step_valid_acc = 0.
                    if total_step % FLAGS.evaluate_every == 0 and total_step > 40000:
                        eval_loss = 0.
                        eval_acc = 0.
                        eval_step = 0
                        for start, end in zip(range(0, dev_sample_size, FLAGS.batch_size), range(FLAGS.batch_size, dev_sample_size, FLAGS.batch_size)):
                            feed_dict = {
                                text_rcnn.input_x: valid_x[start:end],
                                text_rcnn.input_y: valid_y[start:end],
                                text_rcnn.seq_len: valid_len[start:end],
                                text_rcnn.first_stage: False,
                                text_rcnn.dropout_keep_prob: 1.
                            }
                            step, summaries, logits, loss, acc = sess.run([text_rcnn.global_step, dev_summary_op, text_rcnn.logits, text_rcnn.loss_val, text_rcnn.accuracy], feed_dict)
                            dev_summary_writer.add_summary(summaries, step)
                            if all_marked_label_num == 0:
                                labelNumStats(valid_y)
                            zhihuStats(logits, valid_y[start:end])
                            eval_loss += loss
                            eval_acc += acc
                            eval_step += 1
                        this_step_zhihu_score = calZhihuScore()
                        time_str = datetime.datetime.now().isoformat()
                        this_step_valid_acc = eval_acc / eval_step
                        this_step_valid_loss = eval_loss / eval_step
                        print('[%s]Eval_Loss:%.4f\tEval_Accuracy:%.4f\tZhihu_Score:%.4f' % (time_str, this_step_valid_loss, this_step_valid_acc, this_step_zhihu_score))
                    if total_step % FLAGS.checkpoint_every == 0 and total_step > 40000:
                        if not FLAGS.save_best_model:
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print('Saved model checkpoint to %s' % path)
                        elif this_step_zhihu_score > best_valid_zhihu_score:
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print('Saved best zhihu_score model checkpoint to %s[%.4f,%.4f,%.4f]' % (path, this_step_valid_loss, this_step_valid_acc, this_step_zhihu_score))
                            best_valid_zhihu_score = this_step_zhihu_score
                        elif this_step_valid_acc > best_valid_acc:
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print('Saved best acc model checkpoint to %s[%.4f,%.4f,%.4f]' % (path, this_step_valid_loss, this_step_valid_acc, this_step_zhihu_score))
                            best_valid_acc = this_step_valid_acc
                        elif this_step_valid_loss < best_valid_loss:
                            path = saver.save(sess, checkpoint_prefix, global_step=step)
                            print('Saved best loss model checkpoint to %s[%.4f,%.4f,%.4f]' % (path, this_step_valid_loss, this_step_valid_acc, this_step_zhihu_score))
                            best_valid_loss = this_step_valid_loss


right_label_num = 0
right_label_at_pos_num = [0, 0, 0, 0, 0]
sample_num = 0
all_marked_label_num = 0

def labelNumStats(labels):
    global all_marked_label_num
    global sample_num
    all_marked_label_num = np.sum(labels)
    print('all_marked_label_num:' + str(all_marked_label_num))
    sample_num = len(labels)
    print('sample_num:', sample_num)

def zhihuStats(logits, labels, top_predictions = 5):
    global right_label_num
    global right_label_at_pos_num
    predict_index_list = np.argsort(logits)[:, -top_predictions:]
    predict_index_list = predict_index_list[:, [4,3,2,1,0]]
    for predict_labels, marked_labels in zip(predict_index_list, labels):
        for pos, predict_label in zip(range(0, top_predictions), predict_labels):
            if marked_labels[predict_label] > 0.5:
                right_label_num += 1
                right_label_at_pos_num[pos] += 1

def calZhihuScore():
    global right_label_num
    global right_label_at_pos_num
    global sample_num
    global all_marked_label_num
    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)
    recall = float(right_label_num) / all_marked_label_num
    right_label_num = 0
    right_label_at_pos_num = [0, 0, 0, 0, 0]
    if precision + recall == 0.0:
        return 0.0
    return (precision * recall) / (precision + recall)


if __name__ == '__main__':
    tf.app.run()
