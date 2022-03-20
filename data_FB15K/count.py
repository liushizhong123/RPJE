f = open("./train_pra.txt", "r")
num = 0
for line in f:
    num = num + 1
    print(num)
# path 2.txt =1014434
# train.txt = 483142
print(num)
f.close()