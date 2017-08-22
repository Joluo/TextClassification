import re
import numpy as np

def loadSamples(question_train_set, question_topic_train_set, label_map_file_path, eval_sample_path, word2id, valid_rate, num_classes, title_len, sent_len, doc_len, title_only = False, word_freq_threshold = 5):
    samples_x = list()
    samples_title = list()
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
    for line in open(question_train_set).readlines():
        parts = line.strip().split('\t')
        sample_x = list()
        if len(parts) < 3:
            print('bad line:' + line)
            continue
        # handle title
        title_parts = re.split('w23,|w111,|w1570,|w6,|w57,|w4016,|w383,|w54,|w11,|w72,|w25,', parts[2].strip())
        title = []
        desc = []
        for t in title_parts:
            t_parts = t.split(',')
            for x in t_parts:
                if x == '':
                    continue
                if x != '' and len(title) < title_len:
                    title.append(word2id.get(x, 0))
                elif x != '':
                    desc.append(word2id.get(x, 0))
        this_doc_len = 0
        if len(parts) >= 5:
            desc_parts = re.split('w23,|w111,|w1570,|w6,|w57,|w4016,|w383,|w54,|w11,|w72,|w25,', parts[4].strip())
            for d in desc_parts:
                d_parts = d.split(',')
                if len(d_parts) > sent_len:
                    continue
                if len(desc) + len(d_parts) > sent_len:
                    sample_x.append(desc)
                    this_doc_len += 1
                if this_doc_len >= doc_len:
                    break
                for x in d_parts:
                    if x != '':
                        desc.append(word2id.get(x, 0))
            if len(desc) > 0 and this_doc_len < doc_len:
                sample_x.append(desc)
        if len(sample_x) == 0 and len(title):
            continue
        sample_y = []
        for i in question_labels[parts[0]]:
            sample_y.append(label_map[i])
        samples_x.append(sample_x)
        samples_title.append(title)
        samples_y.append(sample_y)
    #print('train_unk_num', train_unk_num)
    #print('ignore_words', ignore_words)
    samples_x = np.array(samples_x)
    samples_y = np.array(samples_y)
    samples_title = np.array(samples_title)
    sample_size = len(samples_x)
    train_sample_size = int(sample_size * (1 - valid_rate))
    return (samples_x[:train_sample_size], samples_title[:train_sample_size], samples_y[:train_sample_size], samples_x[train_sample_size:], samples_title[train_sample_size:], samples_y[train_sample_size:], label_map)

def loadEvalSample(eval_sample_path, word2id, sent_len, doc_len):
    eval_samples = list()
    for line in open(eval_sample_path).readlines():
        parts = line.strip().split('\t')
        sample_x = list()
        if len(parts) < 3:
            print('bad line:' + line)
            continue
        title_parts = re.split('w23,|w111,|w1570,|w6,|w57,|w4016,|w383,|w54,|w11,|w72,|w25,', parts[2].strip())
        title = []
        desc = []
        for t in title_parts:
            t_parts = t.split(',')
            for x in t_parts:
                if x == '':
                    continue
                if x != '' and len(title) < sent_len:
                    title.append(word2id.get(x, 0))
                elif x != '':
                    desc.append(word2id.get(x, 0))
        sample_x.append(title)
        this_doc_len = 1
        if len(parts) >= 5:
            desc_parts = re.split('w23,|w111,|w1570,|w6,|w57,|w4016,|w383,|w54,|w11,|w72,|w25,', parts[4].strip())
            for d in desc_parts:
                d_parts = d.split(',')
                if len(d_parts) > sent_len:
                    continue
                if len(desc) + len(d_parts) > sent_len:
                    sample_x.append(desc)
                    this_doc_len += 1
                if this_doc_len >= doc_len:
                    break
                for x in d_parts:
                    if x != '':
                        desc.append(word2id.get(x, 0))
            if len(desc) > 0 and this_doc_len < doc_len:
                sample_x.append(desc)
        if len(sample_x) == 0:
            continue
        eval_samples.append(sample_x)
    eval_samples = np.array(eval_samples)
    return eval_samples


def loadId2LabelMap(label_map_file_path):
    id2label_map = dict()
    for line in open(label_map_file_path).readlines():
        parts = line.strip().split('\t')
        id2label_map[int(parts[1])] = parts[0]
    return id2label_map

def getTopicEmbedding(top_file_path, w2v_model, word2id, label_map, word_embed_size):
    mem_dict = dict()
    t_cnt = 0
    d_cnt = 0
    for line in open(top_file_path).readlines():
        parts = line.strip().split('\t')
        topic = parts[0]
        #print('topic', topic)
        title = parts[3].split(',')
        desc = list()
        if len(parts) >= 6:
            desc = parts[5].split(',')
        title_vec = list()
        desc_vec = list()
        for w in title:
            w_id = word2id.get(w, 0)
            if w_id > 0:
                title_vec.append(w2v_model[w_id])
                #print(w2v_model[w_id])
        for w in desc:
            w_id = word2id.get(w, 0)
            if w_id > 0:
                desc_vec.append(w2v_model[w_id])
        if len(title_vec) == 0:
            title_vec.append(np.random.uniform(-1,1,size=word_embed_size))
            #title_vec.append(w2v_model[0])
            #print(topic, t_cnt, 'title_none')
            t_cnt += 1
        if len(desc_vec) == 0:
            desc_vec.append(np.random.uniform(-1,1,size=word_embed_size))
            #desc_vec.append(w2v_model[0])
            #print(topic, d_cnt, 'desc_none')
            d_cnt += 1
        title_mean = np.mean(np.array(title_vec), axis=0)
        #print('title_mean', title_mean)
        desc_mean = np.mean(np.array(desc_vec), axis=0)
        merge_mean = title_mean.tolist() + desc_mean.tolist()
        mem_dict[label_map[topic]] = merge_mean
        #overall_vec = list()
        #overall_vec.append(title_mean)
        #overall_vec.append(desc_mean)
        #overall_mean = np.mean(np.array(overall_vec), axis=0).tolist()
        #mem_dict[label_map[topic]] = overall_mean
    mem = list()
    for i in range(1999):
        mem.append(mem_dict[i])
    return np.array(mem)

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
    ws.append([0.0 for i in range(dim)])
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
        #print('vals', vals)
        #mean = np.mean(np.array(vals), axis=0)
        #print('mean', mean)
        #std = np.std(np.array(vals), axis=0)
        #print('std', std)
        #vals = np.array(vals) - mean
        #print('vals_mean', vals)
        #vals = (np.array(vals) - mean) / std
        #vals = vals.tolist()
        #print('vals_std', vals)
        ws.append(vals)
        word2id[ss[0]] = t + 1
    for i in range(dim):
        mv[i] = mv[i] / total
    ws.append([0 for i in range(dim)])
    word2id['unk'] = total + 1
    print("len of ws %s" % len(ws))
    ws.append(mv)
    print("len of ws %s" % len(ws))
    fp.close()
    return np.asarray(ws, dtype=np.float32), word2id


if __name__ == '__main__':
    w2v_model, word2id = load_w2v('/data4/ml/luoqiang/zhihu/data/ieee_zhihu_cup/word_embedding.txt', 256)
    loadSamples('/data4/ml/luoqiang/zhihu/data/ieee_zhihu_cup/question_train_set.txt', '/data4/ml/luoqiang/zhihu/data/ieee_zhihu_cup/question_topic_train_set.txt', '/data4/ml/luoqiang/zhihu/data/samples/id_map', word2id, 0.1, 1999)
