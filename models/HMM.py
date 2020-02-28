import torch
import torch.nn as nn
import torch.nn.functional as F

class HMM(object):
    def __init__(self,N,M):
        """
            HMM模型
        :param N: 状态数，这里对应存在的标注的种类
        :param M: 观测数  这里对应有多少个不同的字
        """
        self.N = N
        self.M = M

        # 初始状态概率 Pi[i]表示初始时刻状态为i的概率
        self.Pi = torch.zeros(N)
        # 状态转移概率矩阵 A[i][j]表示状态从i转移到j的概率
        self.A = torch.zeros(N,N)
        # 观测概率矩阵 B[i][j]表示i状态下生成j观测的概率
        self.B = torch.zeros(N,M)

    def train(self,word_lists,tag_lists,word2id,tag2id):
        """HMM的训练，即根据训练语料对模型参数进行估计,
           因为我们有观测序列以及其对应的状态序列，所以我们
           可以使用极大似然估计的方法来估计隐马尔可夫模型的参数
        参数:
            word_lists: list of list，其中每个元素由字组成的列表，如 ['担','任','科','员']
            tag_lists: list of list，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE'],
                        B表示实体开头，E表示实体结尾，I表示在实体内部，O表示非实体
            word2id: 将字映射为ID
            tag2id: 字典，将标注映射为ID
        """
        assert len(word_lists) == len(tag_lists)

        # 估计状态转移概率矩阵A
        # HMM的一个假设：齐次马尔科夫假设即任意时刻的隐藏状态只依赖以前一个隐藏状态
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len-1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i+1]]
                self.A[current_tagid][next_tagid] += 1
        # smoth
        self.A[self.A == 0.] = 1e-10
        # 计算概率
        self.A = self.A / torch.sum(self.A,dim=1,keepdim=True)

        # 估计观测概率矩阵
        # 观测独立假设，即当前的观测值只依赖以当前的隐藏状态
        for tag_list,word_list in zip(tag_lists,word_lists):
            assert len(tag_list)==len(word_list)
            for tag,word in zip(tag_list,word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B==0.] = 1e-10
        self.B = self.B / torch.sum(self.B,dim=1,keepdim=True)

        # 估计初始概率
        for tag_list in tag_lists:
            init_tagId = tag2id[tag_list[0]]
            self.Pi[init_tagId] += 1
        self.Pi[self.Pi==0] = 1e-10
        self.Pi = self.Pi/self.Pi.sum()

    def decoding(self,word_list,word2id,tag2id):
        """
        使用维特比算法对给定观测序列求隐状态序列， 这里就是对字组成的序列,求其对应的实体标注。
        维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径）
        这时一条路径对应着一个状态序列
        """
        # 解决办法：采用对数概率，这样源空间中的很小概率，就被映射到对数空间的大的负数
        #  同时相乘操作也变成简单的相加操作
        A = torch.log(self.A)
        B = torch.log(self.B)
        Pi = torch.log(self.Pi)

        # 初始化 维比特矩阵viterbi 它的维度为[状态数, 序列长度]
        # 其中viterbi[i, j]表示标注序列的第j个标注为i的所有单个序列(i_1, i_2, ..i_j)出现的概率最大值
        seq_len = len(word_list)
        viterbi = torch.zeros(self.N, seq_len)
        # backpointer是跟viterbi一样大小的矩阵
        # backpointer[i, j]存储的是 标注序列的第j个标注为i时，第j-1个标注的id
        # 等解码的时候，我们用backpointer进行回溯，以求出最优路径
        backpointer = torch.zeros(self.N, seq_len).long()

        # self.Pi[i] 表示第一个字的标记为i的概率
        # Bt[word_id]表示字为word_id的时候，对应各个标记的概率
        # self.A.t()[tag_id]表示各个状态转移到tag_id对应的概率

        # 初始第一步
        start_wordid = word2id.get(word_list[0],None)
        Bt = B.t()  # 转置，[M,N]
        if start_wordid is None:
            # 如果word不在字典里，则假设其状态转移概率分布为均匀分布
            bt1 = (torch.ones(self.N)/self.N).long()
        else:
            bt1 = Bt[start_wordid]
        viterbi[:,0] = self.Pi + bt1
        backpointer[:,0] = -1

        # 递推公式
        # viterbi[tag_id,step] = max(viterbi[:,step-1] * self.A[:,tag_id]) * Bt[word]
        # word是step时刻对应的观测值
        for step in range(1,seq_len):
            wordid = word2id.get(word_list[step],None)
            # bt为时刻t字为wordid时，状态转移的概率
            if wordid is None:
                bt = (torch.ones(self.N)/self.N).long()
            else:
                # 从观测矩阵B中取值
                bt = Bt[wordid]

            for tag_id in range(len(tag2id)):
                # 求前一个step中即维特比矩阵中的前一列每个元素和对应的状态转移矩阵的概率乘积的最大值
                # sigma_(t+1)j = max_i∈(1,N)[sigma_(t)i + a_ij] + b_j(t+1)
                max_prob,max_id = torch.max(viterbi[:,step-1] + A[:,tag_id],dim=0)
                viterbi[tag_id,step] = max_prob + bt[tag_id]
                backpointer[tag_id, step] = max_id

        # 终止， t=seq_len 即 viterbi[:, seq_len]中的最大概率，就是最优路径的概率
        best_path_prob, best_path_pointer = torch.max(viterbi[:, seq_len - 1], dim=0)

        # 回溯，求最优路径
        best_path_pointer = best_path_pointer.item()
        best_path = [best_path_pointer]
        for back_step in range(seq_len-1, 0, -1):
            best_path_pointer = backpointer[best_path_pointer, back_step]
            best_path_pointer = best_path_pointer.item()
            best_path.append(best_path_pointer)

        # 将tag_id组成的序列转化为tag
        assert len(best_path) == len(word_list)
        id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list

    def test(self, word_lists, word2id, tag2id):
        pred_tag_lists = []
        for word_list in word_lists:
            pred_tag_list = self.decoding(word_list, word2id, tag2id)
            pred_tag_lists.append(pred_tag_list)
        return pred_tag_lists
