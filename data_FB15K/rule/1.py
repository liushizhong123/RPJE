f = open("./rule_relation70.txt", "r")
g = open("./rule_relation70_1.txt", "w")
for line in f:
    seg = line.strip().split()
    r1 = seg[0]
    r2 = seg[1]
    pr = seg[2]
    print(r2 + r1 + pr)
    g.write(r2 + "\t" + r1 + "\t" + pr + "\n")

f.close()