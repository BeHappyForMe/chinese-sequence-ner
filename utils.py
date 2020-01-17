import pickle
import torch

def save_model(model,file_name):
    with open(file_name,'wb') as f:
        pickle.dump(model,f)

def flatten_lists(lists):
    """将list of list 压平成list"""
    flatten_list = []
    for list_ in lists:
        if type(list_) == list:
            flatten_list.extend(list_)
        else:
            flatten_list.append(list_)
    return flatten_list

# LSTM模型训练的时候需要在word2id和tag2id加入PAD和UNK
# 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
def extend_maps(word2id, tag2id, for_crf=True):
    word2id['<unk>'] = len(word2id)
    word2id['<pad>'] = len(word2id)
    tag2id['<unk>'] = len(tag2id)
    tag2id['<pad>'] = len(tag2id)
    # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
    if for_crf:
        word2id['<start>'] = len(word2id)
        word2id['<end>'] = len(word2id)
        tag2id['<start>'] = len(tag2id)
        tag2id['<end>'] = len(tag2id)

    return word2id, tag2id

def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False):
    assert len(word_lists) == len(tag_lists)
    for i in range(len(word_lists)):
        word_lists[i].append("<end>")
        if not test:  # 如果是测试数据，就不需要加end token了
            tag_lists[i].append("<end>")

    return word_lists, tag_lists

def sort_by_lengths(word_lists,tag_lists):
    """
        将句子长度排序
    :param word_lists:
    :param tag_lists:
    :return:
    """
    pairs = list(zip(word_lists,tag_lists))
    indices = sorted(range(len(pairs)),key=lambda x:len(pairs[x][0]),reverse=True)

    pairs = [pairs[i] for i in indices]
    word_lists,tag_lists = list(zip(*pairs))
    return word_lists,tag_lists,indices


def tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
    # batch各个元素的长度
    lengths = [len(l) for l in batch]

    return batch_tensor, lengths
