import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class pathTrain(nn.Module):
    def __init__(self, transe, dim=100, d_norm=1, gamma=1):
        super(pathTrain, self).__init__()
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

    def forward(self, trainset, i, e1, e2, rel):
        res = torch.tensor(0).float()
        if len(trainset.fb_path[i]) > 0:
            rel_neg = np.random.randint(trainset.relation_num)
            while trainset.ok.get((e1, rel_neg)) and trainset.ok[(e1, rel_neg)].get(e2):
                rel_neg = np.random.randint(trainset.relation_num)
            # 遍历路径
            for path_id in range(len(trainset.fb_path[i])):
                rel_path = trainset.fb_path[i][path_id][0]
                s = ""
                if trainset.path2s.get(tuple(rel_path)):
                    s = trainset.path2s[tuple(rel_path)]
                else:
                    for ii in range(len(rel_path)):
                        s = s + str(rel_path[ii])
                    trainset.path2s[tuple(rel_path)] = s
                s = trainset.path2s[tuple(rel_path)]
                pr = trainset.fb_path[i][path_id][1]
                pr_path = 0
                if (s, rel) in trainset.path_confidence:
                    pr_path = trainset.path_confidence[(s, rel)]
                pr_path = 0.99 * pr_path + 0.01
                if len(rel_path) > 1:
                    if (rel_path[0], rel_path[1]) in trainset.rule2rel:
                        trainset.rules_used += 1
                        rel_integ = trainset.rule2rel[(rel_path[0], rel_path[1])][0]
                        rel_path[0] = rel_integ
                        rel_path.pop()
                pos_dis = self.calc_path(rel, rel_path)
                neg_dis = self.calc_path(rel_neg, rel_path)

                res += self.calculate_loss(pos_dis, neg_dis, pr * pr_path)
        return res.requires_grad_()

    def calc_path(self, r1, rel_path):
        rel = torch.tensor([r1])
        tmp = torch.zeros(1, 100)
        for i in range(len(rel_path)):
            tmp += self.relation_embedding(torch.tensor([i]))
        return self.relation_embedding(rel) - tmp

    def calculate_loss(self, pos_dis, neg_dis, x):
        distance_diff = self.gamma + x * torch.norm(pos_dis, p=self.d_norm, dim=1) - torch.norm(neg_dis,
                                                                                                p=self.d_norm,
                                                                                                dim=1)
        return torch.sum(F.relu(distance_diff))
