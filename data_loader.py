class trainSet:
    def __init__(self):
        self.ok = {}
        self.relrules_used = 0  # relation pair rule R1 num
        self.rules_used = 0  # path rule R2 num
        self.fb_h = []
        self.fb_l = []
        self.fb_r = []  # ID of the head entity, tail entity and relation
        self.fb_path = []  # all the relation paths
        self.path2s = {}  # path convert to string
        self.path_confidence = {}
        # 实体数量，关系数量
        self.relation_num = 0
        self.entity_num = 0
        self.relation2id = {}
        self.entity2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self.rule2rel = {}  # used for path compositon by R2 rules
        self.rel2rel = {}  # used for relations association by R1 rules
        self.rule_ok = {}
        #  两步关系路径
        self.path = []

    def add(self, x, y, z, path_list):
        # add head entity: x, tail entity: y, relation: z, relation path: path_list, ok: 1 if the triple x-z-y added
        self.fb_h.append(x)
        self.fb_r.append(z)
        self.fb_l.append(y)
        self.fb_path.append(path_list)
        self.ok[(x, z)] = {y: 1}

    def prepare(self):
        f1 = open("./data_FB15K/entity2id.txt", "r")
        f2 = open("./data_FB15K/relation2id.txt", "r")
        for line in f1:
            seg = line.strip().split()
            self.entity2id[seg[0]] = int(seg[1])
            self.id2entity[int(seg[1])] = seg[0]
            self.entity_num += 1
        f1.close()
        for line in f2:
            seg = line.strip().split()
            self.relation2id[seg[0]] = int(seg[1])
            self.id2relation[int(seg[1])] = seg[0]
            self.id2relation[int(seg[1]) + 1345] = "-" + seg[0]
            self.relation_num += 1
        f2.close()

        f_kb = "./data_FB15K/train_pra.txt"
        lines_gen = read(f_kb, 2)
        for line in lines_gen:
            seg1 = line[0].strip().split()
            if seg1[0] not in self.entity2id:
                print("miss entity:" + seg1[0])
            if seg1[1] not in self.entity2id:
                print("miss entity:" + seg1[1])
            e1 = self.entity2id[seg1[0]]
            e2 = self.entity2id[seg1[1]]
            rel = int(seg1[2])
            b = []
            b.clear()
            seg2 = line[1].strip().split()
            for i in range(int(seg2[0])):
                rel_path = []
                rel_path.clear()
                num = int(seg2[1])
                if num == 1:
                    rel_path.append(int(seg2[2]))
                    pr = float(seg2[3])
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    b.append((rel_path, pr))
                if num == 2:
                    rel_path.append(int(seg2[2]))
                    rel_path.append(int(seg2[3]))
                    pr = float(seg2[4])
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    b.append((rel_path, pr))
            self.add(e1, e2, rel, b)
        self.relation_num *= 2

        # f_kb = "./data_FB15K/train_pra1.txt"
        # lines_gen = read(f_kb, 2)
        # for line in lines_gen:
        #     seg1 = line[0].strip().split()
        #     if seg1[0] not in self.entity2id:
        #         print("miss entity:" + seg1[0])
        #     if seg1[1] not in self.entity2id:
        #         print("miss entity:" + seg1[1])
        #     e1 = self.entity2id[seg1[0]]
        #     e2 = self.entity2id[seg1[1]]
        #     rel = int(seg1[2])
        #     b = []
        #     b.clear()
        #     seg2 = line[1].strip().split()
        #     for i in range(int(seg2[0])):
        #         rel_path = []
        #         rel_path.clear()
        #         num = int(seg2[1])
        #         if num == 1:
        #             rel_path.append(int(seg2[2]))
        #             pr = float(seg2[3])
        #             seg2.pop(1)
        #             seg2.pop(1)
        #             seg2.pop(1)
        #             b.append((rel_path, pr))
        #         if num == 3:
        #             rel_path.append(int(seg2[2]))
        #             rel_path.append(int(seg2[3]))
        #             rel_path.append(int(seg2[4]))
        #             pr = float(seg2[5])
        #             seg2.pop(1)
        #             seg2.pop(1)
        #             seg2.pop(1)
        #             seg2.pop(1)
        #             seg2.pop(1)
        #             b.append((rel_path, pr))
        #     self.add(e1, e2, rel, b)
        # self.relation_num *= 2
        print("relation_num=" + str(self.relation_num))
        print("entity_num=" + str(self.entity_num))

        f_confidence = "./data_FB15K/confidence.txt"
        lines = read(f_confidence, 2)
        for line in lines:
            seg1 = line[0].strip().split()
            s = ""
            for i in range(int(seg1[0])):
                s = s + seg1[i + 1]
            seg2 = line[1].strip().split()
            y = -1
            pr = 0.0
            for j in range(int(seg2[0])):
                y = int(seg2[1 + 2 * j])
                pr = float(seg2[2 + 2 * j])
                self.path_confidence[(s, y)] = pr

        print("Load all the R1 rules.")
        f_rule1 = open("./data_FB15K/rule/rule_relation70.txt")
        count_rules1 = 0
        for line in f_rule1:
            seg = line.strip().split()
            rel1 = int(seg[0])
            rel2 = int(seg[1])
            confi = float(seg[2])
            if rel1 in self.rel2rel:
                self.rel2rel[rel1].append((rel2, confi))
            else:
                self.rel2rel[rel1] = [(rel2, confi)]
            self.rule_ok[(rel1, rel2)] = 1
            count_rules1 += 1
        f_rule1.close()

        print("Load all the R2 rules.")
        f_rule2 = open("./data_FB15K/rule/rule_path70.txt")
        count_rules2 = 0
        for line in f_rule2:
            seg = line.strip().split()
            rel1 = int(seg[0])
            rel2 = int(seg[1])
            rel3 = int(seg[2])
            confi = float(seg[3])
            self.rule2rel[(rel1, rel2)] = (rel3, confi)
            count_rules2 += 1
        print("The confidence of rules is: 0.7")
        print("The total number of rules is: " + str((count_rules1 + count_rules2)))
        f_rule2.close()

class testSet:
    def __init__(self):
        self.ok = {}
        self.relrules_used = 0  # relation pair rule R1 num
        self.rules_used = 0  # path rule R2 num
        self.fb_h = []
        self.fb_l = []
        self.fb_r = []  # ID of the head entity, tail entity and relation
        self.fb_path = {}  # all the relation paths
        self.path2s = {}  # path convert to string
        self.path_confidence = {}
        # 实体数量，关系数量
        self.relation_num = 0
        self.entity_num = 0
        self.relation2id = {}
        self.entity2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self.rule2rel = {}  # used for path compositon by R2 rules
        self.rel2rel = {}  # used for relations association by R1 rules
        self.rule_ok = {}
        #  两步关系路径
        self.path = []

    def add_path(self, x, y, z, path_list):
        if z != -1:
            # add head entity: x, tail entity: y, relation: z, relation path: path_list, ok: 1 if the triple x-z-y added
            self.fb_h.append(x)
            self.fb_r.append(z)
            self.fb_l.append(y)
            self.ok[(x, z)] = {y: 1}
        if len(path_list) > 0:
            self.fb_path[(x, y)] = path_list


    def add_triple(self,x,y,z,flag):
        if flag:
            self.fb_h.append(x)
            self.fb_r.append(z)
            self.fb_l.append(y)
            self.ok[(x, z)] = {y: 1}

    def prepare(self):
        print("------------The test process for RPJE_rule!------------\n")
        f1 = open("./data_FB15K/entity2id.txt", "r")
        f2 = open("./data_FB15K/relation2id.txt", "r")
        for line in f1:
            seg = line.strip().split()
            self.entity2id[seg[0]] = int(seg[1])
            self.id2entity[int(seg[1])] = seg[0]
            self.entity_num += 1
        f1.close()
        for line in f2:
            seg = line.strip().split()
            self.relation2id[seg[0]] = int(seg[1])
            self.id2relation[int(seg[1])] = seg[0]
            self.id2relation[int(seg[1]) + 1345] = "-" + seg[0]
            self.relation_num += 1
        f2.close()

        f_kb = "./data_FB15K/test_pra1.txt"
        lines_gen = read(f_kb, 2)
        for line in lines_gen:
            seg1 = line[0].strip().split()
            if seg1[0] not in self.entity2id:
                print("miss entity:" + seg1[0])
            if seg1[1] not in self.entity2id:
                print("miss entity:" + seg1[1])
            e1 = self.entity2id[seg1[0]]
            e2 = self.entity2id[seg1[1]]
            rel = int(seg1[2])
            b = []
            b.clear()
            seg2 = line[1].strip().split()
            for i in range(int(seg2[0])):
                rel_path = []
                rel_path.clear()
                num = int(seg2[1])
                if num == 1:
                    rel_path.append(int(seg2[2]))
                    pr = float(seg2[3])
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    b.append((rel_path, pr))
                if num == 3:
                    rel_path.append(int(seg2[2]))
                    rel_path.append(int(seg2[3]))
                    rel_path.append(int(seg2[4]))
                    pr = float(seg2[5])
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    b.append((rel_path, pr))
            self.add_path(e1, e2, rel, b)

        f_path = "./data_FB15K/path2.txt"
        lines_gen = read(f_path, 2)
        for line in lines_gen:
            seg1 = line[0].strip().split()
            if seg1[0] not in self.entity2id:
                print("miss entity:" + seg1[0])
            if seg1[1] not in self.entity2id:
                print("miss entity:" + seg1[1])
            e1 = self.entity2id[seg1[0]]
            e2 = self.entity2id[seg1[1]]
            b = []
            b.clear()
            seg2 = line[1].strip().split()
            for i in range(int(seg2[0])):
                rel_path = []
                rel_path.clear()
                num = int(seg2[1])
                if num == 1:
                    rel_path.append(int(seg2[2]))
                    pr = float(seg2[3])
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    b.append((rel_path, pr))
                if num == 2:
                    rel_path.append(int(seg2[2]))
                    rel_path.append(int(seg2[3]))
                    pr = float(seg2[4])
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    seg2.pop(1)
                    b.append((rel_path, pr))
            self.add_path(e1, e2, -1, b)

        f_confidence = "./data_FB15K/confidence.txt"
        lines = read(f_confidence, 2)
        for line in lines:
            seg1 = line[0].strip().split()
            s = ""
            for i in range(int(seg1[0])):
                s = s + seg1[i + 1]
            seg2 = line[1].strip().split()
            y = -1
            pr = 0.0
            for j in range(int(seg2[0])):
                y = int(seg2[1 + 2 * j])
                pr = float(seg2[2 + 2 * j])
                self.path_confidence[(s, y)] = pr

        print("Load all the R2 rules.")
        f_rule2 = open("./data_FB15K/rule/rule_path70.txt")
        count_rules2 = 0
        for line in f_rule2:
            seg = line.strip().split()
            rel1 = int(seg[0])
            rel2 = int(seg[1])
            rel3 = int(seg[2])
            confi = float(seg[3])
            self.rule2rel[(rel1, rel2)] = (rel3, confi)
            count_rules2 += 1
        f_rule2.close()
        print("The total number of rules R2 is:" + str(count_rules2))



# 读文件连续的两行
def read(fp: str, n: int):
    i = 0
    lines = []  # a buffer to cache lines

    with open(fp) as f:
        for line in f:
            i += 1
            lines.append(line.strip())  # append a line

            if i >= n:
                yield lines

                # reset buffer
                i = 0
                lines.clear()

    # remaining lines
    if i > 0:
        yield lines


if __name__ == '__main__':
    # train集
    train_data_set = trainSet()
    train_data_set.prepare()
    # test集
    test_data_set = testSet()
    train_data_set.prepare()
