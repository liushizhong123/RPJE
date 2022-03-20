import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda:1')

class EmbedVector(nn.Module):
    def __init__(self, config):
        super(EmbedVector, self).__init__()
        self.config = config
        # 目标维度
        target_size = config.label
        # 实体查找表(14951,100)
        self.entity_embed = nn.Embedding(config.entity_num, config.words_dim)
        # 关系查找表（2690，100）
        self.relation_embed = nn.Embedding(config.relation_num, config.words_dim)
        if not config.train_embed:
            self.embed.weight.requires_grad = False
        if config.qa_mode.upper() == 'LSTM':
            # (100,100,2,0.3)
            self.lstm = nn.LSTM(input_size=config.words_dim,
                                hidden_size=config.hidden_size,
                                num_layers=config.num_layer,
                                dropout=config.rnn_dropout,
                                bidirectional=True)
        elif config.qa_mode.upper() == 'GRU':
            self.gru = nn.GRU(input_size=config.words_dim,
                              hidden_size=config.hidden_size,
                              num_layers=config.num_layer,
                              dropout=config.rnn_dropout,
                              bidirectional=True)
        self.dropout = nn.Dropout(p=config.rnn_fc_dropout)
        self.nonlinear = nn.Tanh()
        # 注意力层
        # self.attn = nn.Sequential(
        #    全连接层
        #    nn.Linear(config.hidden_size * 2 + config.words_dim, config.hidden_size),
        #    self.nonlinear,
        #    nn.Linear(config.hidden_size, 1)
        # )
        self.hidden2tag = nn.Sequential(
            # nn.Linear(config.hidden_size * 2 + config.words_dim, config.hidden_size * 2),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2),
            # 批规范化操作
            nn.BatchNorm1d(config.hidden_size * 2),
            self.nonlinear,
            self.dropout,
            # 最后的线性变换
            nn.Linear(config.hidden_size * 2, target_size)
        )

    def forward(self, x): 
        # x : list (3)
        rel1 = self.relation_embed(torch.tensor([x[0]]).to(device))
        entity_mid = self.entity_embed(torch.tensor([x[1]]).to(device))
        rel2 = self.relation_embed(torch.tensor([x[2]]).to(device))
        path = torch.empty(3,1,100).to(device)
        path[0] = rel1
        path[1] = entity_mid
        path[2] = rel2
        num_word, batch_size, words_dim = path.size()
        # h0 / c0 = (layer*direction, batch_size, hidden_dim)
        if self.config.qa_mode.upper() == 'LSTM':
            outputs, (ht, ct) = self.lstm(path)
        elif self.config.qa_mode.upper() == 'GRU':
            outputs, ht = self.gru(path)
        else:
            print("Wrong Entity Prediction Mode")
            exit(1)
        outputs = outputs.view(-1, outputs.size(2))
        # x = x.view(-1, words_dim)
        # attn_weights = F.softmax(self.attn(torch.cat((x, outputs), 1)), dim=0)
        # attn_applied = torch.bmm(torch.diag(attn_weights[:, 0]).unsqueeze(0), outputs.unsqueeze(0))
        # outputs = torch.cat((x, attn_applied.squeeze(0)), 1)
        tags = self.hidden2tag(outputs).view(num_word, batch_size, -1)
        scores = nn.functional.normalize(torch.mean(tags, dim=0), dim=1)
        # [1,100]
        return scores
