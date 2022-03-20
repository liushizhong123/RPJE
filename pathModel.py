import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class pathTrain(nn.Module):
    def __init__(self, transe, pathEmbedding, device, dim=100, d_norm=1, gamma=1):
        super(pathTrain, self).__init__()
        self.dim = dim
        self.device = device
        self.d_norm = d_norm
        self.gamma = torch.FloatTensor([gamma]).to(self.device)
        self.entity_num = transe.entity_num
        self.relation_num = transe.relation_num
        self.entity_embedding = transe.entity_embedding.to(self.device)
        self.relation_embedding = transe.relation_embedding.to(self.device)
        self.pathEmbedding = pathEmbedding
        # # l <= l / ||l||
        relation_norm = torch.norm(self.relation_embedding.weight.data, dim=1, keepdim=True)
        self.relation_embedding.weight.data = self.relation_embedding.weight.data / relation_norm
        # # e <= e / ||l||
        entity_norm = torch.norm(self.entity_embedding.weight.data, dim=1, keepdim=True)
        self.entity_embedding.weight.data = self.entity_embedding.weight.data / entity_norm

    def forward(self, trainset, i, e1, e2, rel):
        res = torch.tensor(0).float().to(self.device)
        if len(trainset.fb_path[i]) > 0:
            rel_neg = np.random.randint(trainset.relation_num)
            while trainset.ok.get((e1, rel_neg)) and trainset.ok[(e1, rel_neg)].get(e2):
                rel_neg = np.random.randint(trainset.relation_num)
            # 遍历路径
            for path_id in range(len(trainset.fb_path[i])):
                rel_path = trainset.fb_path[i][path_id][0]
                if len(rel_path) > 1:
                    rel_path1 = [rel_path[0],rel_path[2]]
                else:
                    rel_path1 = rel_path
                s = ""
                if trainset.path2s.get(tuple(rel_path1)):
                    s = trainset.path2s[tuple(rel_path1)]
                else:
                    for ii in range(len(rel_path1)):
                        s = s + str(rel_path1[ii])
                    trainset.path2s[tuple(rel_path1)] = s
                s = trainset.path2s[tuple(rel_path1)]
                pr = trainset.fb_path[i][path_id][1]
                pr_path = 0
                if (s, rel) in trainset.path_confidence:
                    pr_path = trainset.path_confidence[(s, rel)]
                pr_path = 0.99 * pr_path + 0.01
                # 2 步路径
                if len(rel_path) > 1:
                    # 可以用规则合并
                    if (rel_path[0], rel_path[2]) in trainset.rule2rel:
                        trainset.rules_used += 1
                        rel_integ = trainset.rule2rel[(rel_path[0], rel_path[2])][0]
                        rel_path[0] = rel_integ
                        # 移除关系路径上的关系和实体
                        rel_path.pop()
                        rel_path.pop()
                        # 计算loss
                        pos_dis = self.calc_path(rel, rel_path)
                        neg_dis = self.calc_path(rel_neg, rel_path)
                        res += self.calculate_loss(pos_dis, neg_dis, pr * pr_path)
                    # 无法用规则导出，用pathEncoder
                    else:
                        # [1,100]
                        rel_encoder = self.pathEmbedding(rel_path)
                        # 计算loss
                        pos_dis = self.calc_path_embed(rel, rel_encoder)
                        neg_dis = self.calc_path_embed(rel_neg, rel_encoder)
                        res += self.calculate_loss(pos_dis, neg_dis, pr * pr_path)
                # 单步路径
                else:
                    pos_dis = self.calc_path(rel, rel_path)
                    neg_dis = self.calc_path(rel_neg, rel_path)
                    res += self.calculate_loss(pos_dis, neg_dis, pr * pr_path)
        return res.requires_grad_()

    def calc_path(self, r1, rel_path):
        rel = torch.tensor([r1]).to(self.device)
        tmp = torch.zeros(1, 100).to(self.device)
        for i in range(len(rel_path)):
            tmp += self.relation_embedding(torch.tensor([i]).to(self.device))
        return self.relation_embedding(rel) - tmp

    def calculate_loss(self, pos_dis, neg_dis, x):
        distance_diff = self.gamma + x * torch.norm(pos_dis, p=self.d_norm, dim=1) - torch.norm(neg_dis,
                                                                                                p=self.d_norm,
                                                                                                dim=1)
        return torch.sum(F.relu(distance_diff))

    def calc_path_embed(self,r1,rel_path):
        rel = torch.tensor([r1]).to(self.device)
        return self.relation_embedding(rel) - rel_path

