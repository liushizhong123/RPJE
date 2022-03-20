import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TranE(nn.Module):
    def __init__(self, entity_num, relation_num, dim=100, d_norm=1, gamma=1):
        """
        :param entity_num: number of entities
        :param relation_num: number of relations
        :param dim: embedding dim
        :param device:
        :param d_norm: measure d(h+l, t), either L1-norm or L2-norm
        :param gamma: margin hyperparameter
        """
        super(TranE, self).__init__()
        self.dim = dim
        self.d_norm = d_norm
        # self.device = device
        self.gamma = torch.FloatTensor([gamma])
        self.entity_num = entity_num
        self.relation_num = relation_num
        self.entity_embedding = nn.Embedding.from_pretrained(
            torch.empty(entity_num, self.dim).uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim)), freeze=False)
        self.relation_embedding = nn.Embedding.from_pretrained(
            torch.empty(relation_num, self.dim).uniform_(-6 / math.sqrt(self.dim), 6 / math.sqrt(self.dim)),
            freeze=False)
        # # l <= l / ||l||
        relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / relation_norm
        # # e <= e / ||l||
        entity_norm = torch.norm(self.entity_embedding.weight.data, dim=1, keepdim=True)
        self.entity_embedding.weight.data = self.entity_embedding.weight.data / entity_norm

    def forward(self, pos_head, pos_tail, pos_relation, neg_head, neg_tail, neg_relation):
        """
        :param pos_head: [1]
        :param pos_relation: [1]
        :param pos_tail: [1]
        :param neg_head: [1]
        :param neg_relation: [1]
        :param neg_tail: [1]
        :return: triples loss
        """
        e1_a = self.entity_embedding(pos_head)
        e2_a = self.entity_embedding(pos_tail)
        rel_a = self.relation_embedding(pos_relation)
        e1_b = self.entity_embedding(neg_head)
        e2_b = self.entity_embedding(neg_tail)
        rel_b = self.relation_embedding(neg_relation)

        pos_dis = e1_a + rel_a - e2_a
        neg_dis = e1_b + rel_b - e2_b
        # return pos_head_and_relation, pos_tail, neg_head_and_relation, neg_tail
        # 保存梯度，以便之后的梯度更新，梯度更新必须返回　loss　为　float
        return self.calculate_loss(pos_dis, neg_dis).requires_grad_()

    def calculate_loss(self, pos_dis, neg_dis):
        """
        :param pos_dis: [embed_dim]
        :param neg_dis: [embed_dim]
        :return: triples loss: [1]
        """
        distance_diff = self.gamma + torch.norm(pos_dis, p=self.d_norm, dim=1) - torch.norm(neg_dis, p=self.d_norm,
                                                                                            dim=1)
        return torch.sum(F.relu(distance_diff))
