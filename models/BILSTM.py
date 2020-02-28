import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM(nn.Module):
    def __init__(self,vocab_size,emb_size,hidden_size,out_size,dropout=0.1):
        super(BiLSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size,emb_size)
        self.bilstm = nn.LSTM(emb_size,hidden_size,batch_first=True,bidirectional=True)
        self.fc = nn.Linear(2*hidden_size,out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,lengths):
        # [b,l,emb_size ]
        emb = self.dropout(self.embedding(x))
        # 这里要求输入按长度递减排好序，否则enforce_sorted设置为false,低版本方法有不同之处
        emb = nn.utils.rnn.pack_padded_sequence(emb,lengths,batch_first=True)
        emb,_ = self.bilstm(emb)
        emb,_ = nn.utils.rnn.pad_packed_sequence(emb,batch_first=True,padding_value=0.,total_length=x.shape[1])
        scores = self.fc(emb)

        return scores

    def test(self, x, lengths, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(x, lengths)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids

def cal_loss(logits, targets, tag2id):
    """计算损失
    参数:
        logits: [B, L, out_size]
        targets: [B, L]
        lengths: [B]
    """
    PAD = tag2id.get('<pad>')
    assert PAD is not None

    # TODO Be Careful
    mask = (targets != PAD)  # [B, L]
    # print(type(mask))
    targets = targets[mask]   # 拉平成了一个维度 B * L (去除了mask中为false的，实际长度要减去mask中为false值)
    # print(targets.shape)
    out_size = logits.size(2)
    # mask.unsqueeze(2) 【 B, L, 1 】
    # expand把第三维度复制成outsize
    logits = logits.masked_select(
        mask.unsqueeze(2).expand(-1, -1, out_size)
    ).contiguous().view(-1, out_size)
    # 最后输出维度为【B*L,outsize】
    # 第一维度其实减去了mask掉的

    assert logits.size(0) == targets.size(0)
    loss = F.cross_entropy(logits, targets)

    return loss

# target = torch.randn(16,10)
# target[:,6:] = 0
# mask = target != 0
# target = target[mask]
# print(target.shape)
# logits = torch.randn(16,10,8)
# re = logits.masked_select(mask.unsqueeze(2).expand(-1,-1,8)).contiguous()
# print(re.shape)