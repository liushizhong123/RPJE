import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ruleTrain(nn.Module):
    def __init__(self, transe, dim=100, d_norm=1, gamma=1):
        super(ruleTrain, self).__init__()
        self.dim = dim
        # self.device = device
        self.d_norm = d_norm
        self.gamma = torch.FloatTensor([gamma])
        self.entity_num = transe.entity_num
        self.relation_num = transe.relation_num
        self.entity_embedding = transe.entity_embedding
        self.relation_embedding = transe.relation_embedding
        # # l <= l / ||l||
        relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / relation_norm
        # # e <= e / ||l||
        entity_norm = torch.norm(self.entity_embedding.weight.data, dim=1, keepdim=True)
        self.entity_embedding.weight.data = self.entity_embedding.weight.data / entity_norm

    def forward(self, rel_a, trainset):
        res = torch.tensor(0).float()
        if rel_a in trainset.rel2rel:
            for i in range(len(trainset.rel2rel[rel_a])):
                rel_rpos = trainset.rel2rel[rel_a][i][0]
                rel_pconfi = trainset.rel2rel[rel_a][i][1]
                pos_dis = self.calc_rule(rel_a, rel_rpos)

                rel_rneg = np.random.randint(trainset.relation_num)
                while (rel_a, rel_rneg) in trainset.rule_ok:
                    rel_rneg = np.random.randint(trainset.relation_num)
                neg_dis = self.calc_rule(rel_a, rel_rneg)

                trainset.relrules_used += 1

                res += self.calculate_loss(pos_dis, neg_dis, rel_pconfi)
        return res.requires_grad_()

    def calculate_loss(self, pos_dis, neg_dis, rel_pconfi):
        """
        :param rel_pconfi:
        :param pos_dis: [embed_dim]
        :param neg_dis: [embed_dim]
        :return: triples loss: [1]
        """
        distance_diff = self.gamma + rel_pconfi * torch.norm(pos_dis, p=self.d_norm, dim=1) - torch.norm(neg_dis,
                                                                                                         p=self.d_norm,
                                                                                                         dim=1)
        return torch.sum(F.relu(distance_diff))

    def calc_rule(self, rel_a, rel_rpos):
        dis = self.relation_embedding(torch.tensor([rel_a])) - self.relation_embedding(
            torch.tensor([rel_rpos]))
        return dis
