import random
import numpy as np
import torch
from torch import optim
from Transe import TranE
from data_loader import trainSet
from RuleModel import ruleTrain
from PathEncoder import pathTrain
import logging

from embedding import EmbedVector
from embeddingConfig import Config

version = ""


def main():
    logging.basicConfig(level=logging.INFO, filename='./mylog/train.log', filemode='w')
    logging.info("Start to prepare!")
    trainset = trainSet()
    trainset.prepare()
    logging.info("Prepare Success!")
    logging.info("Start Training!")
    # 嵌入维度
    n = 100
    rate = 0.001
    logging.info(f'n : {str(n)} rate : {str(rate)} ')
    # training procedure
    margin = 1
    margin_rel = 1
    logging.info(f'margin : {str(margin)}   margin_rel : {str(margin_rel)}')
    nbatches = 100
    nepoches = 100
    logging.info(f'nbatches : {str(nbatches)}')
    logging.info(f'nepoches : {str(nepoches)}')
    # batch size
    batchsize = 128
    logging.info(f' The total number of triples is : {str(len(trainset.fb_h))}')
    logging.info(f' batchsize is : {str(batchsize)}')
    entity_num = trainset.entity_num
    relation_num = trainset.relation_num

    config = Config(trainset)

    # 创建transe模型
    transe = TranE(trainset.entity_num, trainset.relation_num, dim=n, d_norm=1,
                   gamma=margin)
    # 规则训练模型
    ruletrain = ruleTrain(transe,dim=n, d_norm=1, gamma=margin)

    # 路径编码模型
    # pathEmbedding = EmbedVector(config).to(device)

    # 路径训练模型
    # pathtrain = pathTrain(transe, pathEmbedding, device, dim=n, d_norm=1, gamma=margin).to(device)
    pathtrain = pathTrain(transe, dim=n, d_norm=1, gamma=margin)

    # 优化器
    optimizer1 = optim.SGD(transe.parameters(), lr=rate)
    optimizer2 = optim.SGD(ruletrain.parameters(), lr=rate)
    optimizer3 = optim.SGD(pathtrain.parameters(), lr=rate)

    # 开始训练
    # epoch process
    for epoch in range(nepoches):
        # loss
        res = 0
        for batch in range(nbatches):
            for k in range(batchsize):
                # random select a negative entity
                entity_neg = np.random.randint(entity_num)
                i = np.random.randint(len(trainset.fb_h))
                e1 = trainset.fb_h[i]
                rel = trainset.fb_r[i]
                e2 = trainset.fb_l[i]
                rand_tmp = random.randint(0, 99)
                if rand_tmp < 25:
                    while trainset.ok[(e1, rel)].get(entity_neg):
                        entity_neg = np.random.randint(entity_num)
                    # 计算L1loss
                    e1_cuda = torch.tensor([e1])
                    e2_cuda = torch.tensor([e2])
                    rel_cuda = torch.tensor([rel])
                    entity_neg_cuda = torch.tensor([entity_neg])
                    loss1 = transe(e1_cuda, e2_cuda, rel_cuda, e1_cuda, entity_neg_cuda, rel_cuda)
                    res += loss1
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()

                    # 计算L2 loss
                    ruletrain.relation_embedding = transe.relation_embedding
                    ruletrain.entity_embedding = transe.entity_embedding
                    loss2 = ruletrain(rel, trainset)
                    res += loss2
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()

                    # 计算L3 loss
                    pathtrain.relation_embedding = ruletrain.relation_embedding
                    pathtrain.entity_embedding = ruletrain.entity_embedding
                    # pathEmbedding.relation_embed = ruletrain.relation_embedding
                    # pathEmbedding.entity_embed = ruletrain.entity_embedding
                    loss3 = 3 * pathtrain(trainset, i, e1, e2, rel)
                    res += loss3 
                    optimizer3.zero_grad()
                    loss3.backward()
                    optimizer3.step()

                elif rand_tmp < 50:
                    while trainset.ok.get((entity_neg, rel)) and trainset.ok[(entity_neg, rel)].get(e2):
                        entity_neg = np.random.randint(entity_num)
                    # 计算L1loss
                    e1_cuda = torch.tensor([e1])
                    e2_cuda = torch.tensor([e2])
                    rel_cuda = torch.tensor([rel])
                    entity_neg_cuda = torch.tensor([entity_neg])
                    loss1 = transe(e1_cuda, e2_cuda,
                                   rel_cuda, entity_neg_cuda,
                                   e2_cuda, rel_cuda)
                    res += loss1
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()

                    # 计算L2 loss
                    ruletrain.relation_embedding = transe.relation_embedding
                    ruletrain.entity_embedding = transe.entity_embedding
                    loss2 = ruletrain(rel, trainset)
                    res += loss2
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()

                    # 计算L3 loss
                    pathtrain.relation_embedding = ruletrain.relation_embedding
                    pathtrain.entity_embedding = ruletrain.entity_embedding
                    # pathEmbedding.relation_embed = ruletrain.relation_embedding
                    # pathEmbedding.entity_embed = ruletrain.entity_embedding
                    loss3 = 3 * pathtrain(trainset, i, e1, e2, rel)
                    res += loss3
                    optimizer3.zero_grad()
                    loss3.backward()
                    optimizer3.step()
                else:
                    rel_neg = np.random.randint(relation_num)
                    while trainset.ok.get((e1, rel_neg)) and trainset.ok[(e1, rel_neg)].get(e2):
                        rel_neg = np.random.randint(relation_num)
                    # 计算L1loss
                    e1_cuda = torch.tensor([e1])
                    e2_cuda = torch.tensor([e2])
                    rel_cuda = torch.tensor([rel])
                    rel_neg_cuda = torch.tensor([rel_neg])
                    loss1 = transe(e1_cuda, e2_cuda, rel_cuda, e1_cuda, e2_cuda, rel_neg_cuda)
                    res += loss1
                    optimizer1.zero_grad()
                    loss1.backward()
                    optimizer1.step()

                    # 计算L2 loss
                    ruletrain.relation_embedding = transe.relation_embedding
                    ruletrain.entity_embedding = transe.entity_embedding
                    loss2 = ruletrain(rel, trainset)
                    res += loss2
                    optimizer2.zero_grad()
                    loss2.backward()
                    optimizer2.step()

                    # 计算L3 loss
                    pathtrain.relation_embedding = ruletrain.relation_embedding
                    pathtrain.entity_embedding = ruletrain.entity_embedding
                    # pathEmbedding.relation_embed = ruletrain.relation_embedding
                    # pathEmbedding.entity_embed = ruletrain.entity_embedding
                    loss3 = 3 * pathtrain(trainset, i, e1, e2, rel)
                    res += loss3
                    optimizer3.zero_grad()
                    loss3.backward()
                    optimizer3.step()

        # e <= e / ||e||
        entity_norm = torch.norm(pathtrain.entity_embedding.weight.data, dim=1, keepdim=True)
        pathtrain.entity_embedding.weight.data = pathtrain.entity_embedding.weight.data / entity_norm
        # # l <= l / ||l||
        relation_norm = torch.norm(pathtrain.relation_embedding.weight.data, dim=1, keepdim=True)
        pathtrain.relation_embedding.weight.data = pathtrain.relation_embedding.weight.data / relation_norm

        # 更新嵌入
        transe.entity_embedding = pathtrain.entity_embedding
        transe.relation_embedding = pathtrain.relation_embedding
        # ruletrain.relation_embedding = pathtrain.relation_embedding
        # ruletrain.entity_embedding = pathtrain.entity_embedding
        # pathEmbedding.entity_embed = pathtrain.entity_embedding
        # pathEmbedding.relation_embed = pathtrain.relation_embedding

        logging.info(f' epoch : {str(epoch)}  loss : {str(res)}')
        logging.info(f' The number of R2 rules (rules of length 2) used in this epoch is : {str(trainset.rules_used)}')
        logging.info(f' The number of R1 rules (rules of length 1) used in this epoch is : {str(trainset.relrules_used)}')
        if epoch > 100 and (epoch + 1) % 100 == 0:
            save_n = str((epoch + 1) // 100)
            f2 = open("./data_FB15K/res4/relation2vec_rule70_" + save_n + ".txt", "w")
            f3 = open("./data_FB15K/res4/entity2vec_rule70_" + save_n + ".txt", "w")
            # 输出去 cuda 化
            relation_vec = pathtrain.relation_embedding.weight.data.detach().cpu().numpy()
            entity_vec = pathtrain.entity_embedding.weight.data.detach().cpu().numpy()
            for i in range(relation_num):
                for ii in range(n):
                    f2.write(str(relation_vec[i][ii]) + "\t")
                f2.write("\n")
            for i in range(entity_num):
                for ii in range(n):
                    f3.write(str(entity_vec[i][ii]) + "\t")
                f3.write("\n")
            f2.close()
            f3.close()
            logging.info("Saving the training result succeed!")
    logging.info("Training finished.")


if __name__ == '__main__':
    main()
