import lazyllm

test1 = lambda a: a + 1
test2 = lambda a: a * 4
test3 = lambda a: a / 2

prl1 = lazyllm.warp(test1, test2, test3)
# prl2 = lazyllm.warp(path1=test1, path2=test2, path3=test3).asdict # Not Implemented
prl3 = lazyllm.warp(test1, test2, test3).astuple
prl4 = lazyllm.warp(test1, test2, test3).aslist
prl5 = lazyllm.warp(test1, test2, test3).join('，')

inputs = [1, 2, 3]

print("默认输出：prl1(1) -> ", prl1(inputs), type(prl1(inputs)))
print("输出元组：prl3(1) -> ", prl3(inputs), type(prl3(inputs)))
print("输出列表：prl4(1) -> ", prl4(inputs), type(prl4(inputs)))
print("输出字符串：prl5(1) -> ", prl5(inputs), type(prl5(inputs)))