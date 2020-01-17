import torch
class HMM1(object):
    def __init__(self,N,M):
        self.N =N
        self.M = M

        self.Pi = torch.zeros(self.N)
        self.A = torch.zeros(self.N,self.N)
        self.B = torch.zeros(self.N,self.M)

    def train(self, word_lists, tag_lists, word2id, tag2id):
        """HMM的训练，即根据训练语料对模型参数进行估计,
           因为我们有观测序列以及其对应的状态序列，所以我们
           可以使用极大似然估计的方法来估计隐马尔可夫模型的参数
        参数:
            word_lists: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
            tag_lists: 列表，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
            word2id: 将字映射为ID
            tag2id: 字典，将标注映射为ID
        """

        assert len(tag_lists) == len(word_lists)

        # 估计转移概率矩阵
        for tag_list in tag_lists:
            seq_len = len(tag_list)
            for i in range(seq_len - 1):
                current_tagid = tag2id[tag_list[i]]
                next_tagid = tag2id[tag_list[i+1]]
                self.A[current_tagid][next_tagid] += 1
        # 问题：如果某元素没有出现过，该位置为0，这在后续的计算中是不允许的
        # 解决方法：我们将等于0的概率加上很小的数
        self.A[self.A == 0.] = 1e-10
        self.A = self.A / self.A.sum(dim=1, keepdim=True)

        # 估计观测概率矩阵
        for tag_list, word_list in zip(tag_lists, word_lists):
            assert len(tag_list) == len(word_list)
            for tag, word in zip(tag_list, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(dim=1, keepdim=True)

        # 估计初始状态概率
        for tag_list in tag_lists:
            init_tagid = tag2id[tag_list[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()


    def decoding(self,word_list,tag2id,word2id):

        Pi = torch.log(self.Pi)
        A = torch.log(self.A)
        B = torch.log(self.B)

        seq_len = len(word_list)
        viterbi = torch.zeros(self.N,seq_len)
        backpointer = torch.zeros(self.N,seq_len).int()

        # 第一步
        startwordid = word2id.get(word_list[0],None)
        # Bt = B.t()
        if startwordid is None:
            bt = torch.log(torch.ones(self.N)/self.N)
        else:
            bt = B[:,startwordid]
        viterbi[:,0] = Pi + bt
        backpointer[:,0] = -1

        for step in range(1,seq_len):
            curwordid = word2id.get(word_list[step],None)
            if curwordid is None:
                bt = torch.log(torch.ones(self.N)/self.N)
            else:
                bt = B[:,curwordid]
            for tag_id in range(len(tag2id)):
                max_prob,max_id =torch.max(viterbi[:,step-1]+A[:,tag_id],dim=0)
                # TODO
                viterbi[tag_id,step] = max_prob + bt[tag_id]
                backpointer[tag_id,step] = max_id

        # 求最后一步得出的一列viterbi求最大后回溯
        # best_path_prob为最优路径的概率值
        best_path_prob,best_path_point = torch.max(viterbi[:,seq_len-1],dim=0)

        # 回溯最优路径
        best_path_point = int(best_path_point.item())
        best_path = [best_path_point]
        for back_step in range(seq_len-1,0,-1):
            best_path_point = backpointer[best_path_point,back_step]
            best_path_point = best_path_point.item()
            best_path.append(best_path_point)

        # 将路径id转为tag
        assert len(word_list) == len(best_path)
        id2tag = dict((id_,tag) for tag,id_ in tag2id.items())
        tag_list = [id2tag[id_] for id_ in reversed(best_path)]

        return tag_list

    def test(self,word_lists,word2id,tag2id):
        pred_tag_lists = []
        for word_list in word_lists:
            tag_list = self.decoding(word_list,tag2id,word2id)
            pred_tag_lists.append(tag_list)
        return pred_tag_lists
