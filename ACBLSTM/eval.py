import tensorflow as tf
import numpy as np
from tflearn.data_utils import pad_sequences
from tensorflow.contrib import learn
from data_helper import load_w2v, loadEvalSample, loadId2LabelMap

tf.app.flags.DEFINE_string('eval_data_file', 'zhihu/data/samples/merge_eval_samples', 'eval data sample')
tf.app.flags.DEFINE_string('raw_eval_file', 'zhihu/data/ieee_zhihu_cup/question_eval_set.txt', 'raw eval file, extract question id')
tf.app.flags.DEFINE_string('prediction_file', 'zhihu/data/res/acblstm.csv', 'predicted csv file.')
tf.app.flags.DEFINE_string('word_embedding_file', 'zhihu/data/ieee_zhihu_cup/word_embedding.txt', 'word embedding file')
tf.app.flags.DEFINE_string('label_map', 'zhihu/data/samples/id_map', 'label id mapping')
tf.app.flags.DEFINE_string('checkpoint_dir', 'zhihu/TextCNN/runs/1497513884/checkpoints_tmp', 'Checkpoint directory from training run')
tf.app.flags.DEFINE_integer('sample_len', 150, 'max sample len')
tf.app.flags.DEFINE_integer('batch_size', 512, 'Batch Size')
tf.app.flags.DEFINE_integer('top_predictions', 5, 'Top n predicted labels')
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
tf.app.flags.DEFINE_boolean('debug', False, 'Debug mode')

FLAGS = tf.app.flags.FLAGS

def main(_):
    w2v_model, word2id = load_w2v(FLAGS.word_embedding_file, 256)
    eval_x = loadEvalSample(FLAGS.eval_data_file, word2id)
    eval_x = pad_sequences(eval_x, maxlen=FLAGS.sample_len, value = 0.)
    eval_sample_size = len(eval_x)
    print('Evaluation samples size:' + str(eval_sample_size))
    id2label_map = loadId2LabelMap(FLAGS.label_map)

    checkpoint_file = 'zhihu/ACBLSTM/runs/1499140857/checkpoints/model-117600'#tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    print('checkpoint_file:' + checkpoint_file)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default(), tf.device('/gpu:0'):
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            rnn_input_dropout_keep_prob = graph.get_operation_by_name('rnn_input_dropout_keep_prob').outputs[0]
            rnn_output_dropout_keep_prob = graph.get_operation_by_name('rnn_output_dropout_keep_prob').outputs[0]
            phase_train = graph.get_operation_by_name('phase_train').outputs[0]
            logits = graph.get_operation_by_name('logits').outputs[0]
            all_predictions = []
            all_scores = []
            end_pos = 0
            for start, end in zip(range(0, eval_sample_size, FLAGS.batch_size), range(FLAGS.batch_size, eval_sample_size, FLAGS.batch_size)):
                #print(eval_x)
                feed_dict = {
                    input_x: eval_x[start:end],
                    rnn_input_dropout_keep_prob: 1.,
                    rnn_output_dropout_keep_prob: 1.,
                    phase_train: False
                }
                batch_logits = sess.run(logits, feed_dict)
                batch_predictions, batch_scores = getTopNPredictions(batch_logits, id2label_map, FLAGS.top_predictions)
                all_predictions.extend(batch_predictions)
                all_scores.extend(batch_scores)
                end_pos = end
            if end_pos < eval_sample_size:
                feed_dict = {
                    input_x: eval_x[end:],
                    rnn_input_dropout_keep_prob: 1.,
                    rnn_output_dropout_keep_prob: 1.,
                    phase_train: False
                }
                batch_logits = sess.run(logits, feed_dict)
                batch_predictions, batch_scores = getTopNPredictions(batch_logits, id2label_map, FLAGS.top_predictions)
                all_predictions.extend(batch_predictions)
                all_scores.extend(batch_scores)
            print(len(all_predictions))
    cnt = 0
    id2question_map = dict()
    for line in open(FLAGS.raw_eval_file).readlines():
        parts = line.strip().split('\t')
        id2question_map[cnt] = parts[0]
        cnt += 1
    fp = open(FLAGS.prediction_file, 'w')
    for i in range(len(all_predictions)):
        res_str = id2question_map[i]
        for j in range(len(all_predictions[i])):
            if FLAGS.debug:
                res_str += ',' + all_predictions[i][j] + ':' + str(all_scores[i][j])
            else:
                res_str += ',' + all_predictions[i][j]
        fp.write(res_str + '\n')
        cnt += 1
    fp.flush()
    fp.close()

def getTopNPredictions(logits, id2label_map, top_predictions):
    #print('logits')
    #print(logits)
    index_list = np.argsort(logits)[:, -top_predictions:]
    index_list = index_list[:, [4,3,2,1,0]]
    label_list = []
    score_list = []
    cnt = 0
    for index in index_list:
        labels = []
        scores = []
        for i in index:
            label=id2label_map[i]
            labels.append(label)
            scores.append(logits[cnt, i])
        label_list.append(labels)
        score_list.append(scores)
        cnt += 1
    return label_list, score_list

if __name__ == '__main__':
    tf.app.run()
