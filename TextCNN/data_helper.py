
import numpy as np

def loadSamples(question_train_set, question_topic_train_set, label_map_file_path, word2id, valid_rate, num_classes, title_only = False):
    samples_x = list()
    samples_y = list()
    label_map = dict()
    for line in open(label_map_file_path).readlines():
        parts = line.strip().split('\t')
        label_map[parts[0]] = int(parts[1])
    question_labels = dict()
    for line in open(question_topic_train_set).readlines():
        parts = line.strip().split('\t')
        if len(parts) != 2:
            print('Warning line: ' + line)
            question_labels[parts[0]] = list()
            continue
        question_labels[parts[0]] = parts[1].split(',')
    sample_y_tmp = [0 for i in range(num_classes)]
    for line in open(question_train_set).readlines():
        parts = line.strip().split('\t')
        sample_x = list()
        if len(parts) < 3:
            print('bad line:' + line)
            continue
        for w in parts[2].strip().split(','):
        #for w in parts[1].strip().split(','):
            sample_x.append(word2id.get(w, 0))
        if not title_only and len(parts) >= 5:
            for w in parts[4].strip().split(','):
            #for w in parts[3].strip().split(','):
                sample_x.append(word2id.get(w, 0))
        if len(sample_x) == 0:
            continue
        sample_y = sample_y_tmp[:]
        for i in question_labels[parts[0]]:
            sample_y[label_map[i]] = 1
        samples_x.append(sample_x)
        samples_y.append(sample_y)
    samples_x = np.array(samples_x)
    samples_y = np.array(samples_y)
    sample_size = len(samples_x)
    train_sample_size = int(sample_size * (1 - valid_rate))
    return (samples_x[:train_sample_size], samples_y[:train_sample_size], samples_x[train_sample_size:], samples_y[train_sample_size:])

def loadEvalSample(eval_sample_path, word2id):
    eval_samples = list()
    for line in open(eval_sample_path).readlines():
        words = line.strip().split(' ')
        sample_ids = list()
        for w in words:
            sample_ids.append(word2id.get(w, 0))
        eval_samples.append(sample_ids)
    eval_samples = np.array(eval_samples)
    return eval_samples


def loadId2LabelMap(label_map_file_path):
    id2label_map = dict()
    for line in open(label_map_file_path).readlines():
        parts = line.strip().split('\t')
        id2label_map[int(parts[1])] = parts[0]
    return id2label_map


def load_w2v(path, embedding_size):
    fp = open(path, "r")
    word2id = dict()
    print("load word2vec model from:", path)
    line = fp.readline().strip()
    ss = line.split(" ")
    total = int(ss[0])
    dim = int(ss[1])
    assert (dim == embedding_size)
    ws = []
    mv = [0 for i in range(dim)]
    # The first for 0
    ws.append([0 for i in range(dim)])
    print("total %s words" % total)
    for t in range(total):
        line = fp.readline().strip()
        ss = line.split(" ")
        assert (len(ss) == (dim + 1))
        vals = []
        for i in range(1, dim + 1):
            fv = float(ss[i])
            mv[i - 1] += fv
            vals.append(fv)
        ws.append(vals)
        word2id[ss[0]] = t + 1
    for i in range(dim):
        mv[i] = mv[i] / total
    print("len of ws %s" % len(ws))
    ws.append(mv)
    print("len of ws %s" % len(ws))
    fp.close()
    return np.asarray(ws, dtype=np.float32), word2id


if __name__ == '__main__':
    w2v_model, word2id = load_w2v('/data4/ml/luoqiang/zhihu/data/ieee_zhihu_cup/word_embedding.txt', 256)
    loadSamples('/data4/ml/luoqiang/zhihu/data/ieee_zhihu_cup/question_train_set.txt', '/data4/ml/luoqiang/zhihu/data/ieee_zhihu_cup/question_topic_train_set.txt', '/data4/ml/luoqiang/zhihu/data/samples/id_map', word2id, 0.1, 1999)
