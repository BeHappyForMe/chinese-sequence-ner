import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BILSTM import BiLSTM,cal_loss
from models.BILSTM_CRF import BiLSTM_CRF,cal_lstm_crf_loss
from config import TrainingConfig,LSTMConfig
from utils import sort_by_lengths,tensorized

from copy import deepcopy
from tqdm import tqdm,trange


class BiLSTM_operator(object):
    def __init__(self,vocab_size,out_size,crf=True):
        """
            训练与测试LSTM模型的操作类
        :param vocab_size: 词典大小
        :param out_size: 标注种类
        :param crf: 是否添加crf层
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 模型参数
        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size

        # 加载不同的model
        self.crf = crf
        if crf:
            self.model = BiLSTM_CRF(vocab_size,self.emb_size,self.hidden_size,out_size).to(self.device)
            self.cal_loss_func = cal_lstm_crf_loss
        else:
            self.model = BiLSTM(vocab_size,self.emb_size,self.hidden_size,out_size).to(self.device)
            self.cal_loss_func = cal_loss

        # 加载训练参数：
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        #初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr)
        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None

    def train(self,word_lists,tag_lists,dev_word_lists,dev_tag_lists,word2id,tag2id):
        #对数据集按长度排序
        word_lists,tag_lists,_ = sort_by_lengths(word_lists,tag_lists)
        dev_word_lists,dev_tag_lists,_ = sort_by_lengths(dev_word_lists,dev_tag_lists)

        print("训练数据总量:{}".format(len(word_lists)))

        batch_size = self.batch_size
        epoch_iterator = trange(1,self.epoches+1,desc="Epoch")
        for epoch in epoch_iterator:
            self.step = 0
            losses = 0.
            for idx in trange(0,len(word_lists),batch_size,desc="Iteration"):
                batch_sents = word_lists[idx:idx+batch_size]
                batch_tags = tag_lists[idx:idx+batch_size]
                losses += self.train_step(batch_sents,batch_tags,word2id,tag2id)

                if self.step%TrainingConfig.print_step == 0:
                    total_step = (len(word_lists)//batch_size + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        epoch, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.
            # 每轮结束测试在验证集上的性能，保存最好的一个
            val_loss = self.validate(
                        dev_word_lists, dev_tag_lists, word2id, tag2id)
            print("Epoch {}, Val Loss:{:.4f}".format(epoch, val_loss))

    def train_step(self,batch_sents,batch_tags,word2id,tag2id):
        self.model.train()
        self.step+=1

        # 数据转tensor
        tensorized_sents,lengths = tensorized(batch_sents,word2id)
        targets,_ = tensorized(batch_tags,tag2id)
        tensorized_sents,targets = tensorized_sents.to(self.device),targets.to(self.device)

        #forward  【 batch,  seq_len,  oupsize 】
        scores = self.model(tensorized_sents,lengths)

        # 计算损失，反向传递
        self.model.zero_grad()
        loss = self.cal_loss_func(scores,targets,tag2id)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_tag_lists, word2id, tag2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_tags = dev_tag_lists[ind:ind+self.batch_size]
                tensorized_sents, lengths = tensorized(batch_sents, word2id)
                tensorized_sents = tensorized_sents.to(self.device)
                targets, lengths = tensorized(batch_tags, tag2id)
                targets = targets.to(self.device)

                # forward
                scores = self.model(tensorized_sents, lengths)

                # 计算损失
                loss = self.cal_loss_func(scores, targets, tag2id).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self,word_lists,tag_lists,word2id,tag2id):
        """返回最佳模型在测试集上的预测结果"""
        # 数据准备
        word_lists,tag_lists,indices = sort_by_lengths(word_lists,tag_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(tensorized_sents,lengths,tag2id)

        # 将id转化为标注
        pred_tag_lists = []
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        for i, ids in enumerate(batch_tagids):
            tag_list = []
            if self.crf:
                for j in range(lengths[i] - 1):  # crf解码过程中，end被舍弃
                    tag_list.append(id2tag[ids[j].item()])
            else:
                for j in range(lengths[i]):
                    tag_list.append(id2tag[ids[j].item()])
            pred_tag_lists.append(tag_list)

        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [tag_lists[i] for i in indices]

        return pred_tag_lists, tag_lists




labels = torch.randn(10,5)
logits = torch.randn(10,5,3)
attention_mask = torch.ones(10,5)
attention_mask[:,3:]=0
print(attention_mask)
print("----"*20)
active_loss = attention_mask.view(-1) == 1
print(active_loss)
active_logits = logits.view(-1, 3)[active_loss]
print(active_logits.shape)
active_labels = labels.view(-1)[active_loss]
print(active_labels.shape)

