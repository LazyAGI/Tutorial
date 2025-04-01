import lazyllm

cond = lambda x: x > 0
true_path = lambda x: x * 2
false_path = lambda x: -x

ifs_flow = lazyllm.ifs(cond, true_path, false_path)

res1 = ifs_flow(10)
print('输入：10，输出：', res1)
res2 = ifs_flow(-5)
print('输入：-5，输出：', res2)