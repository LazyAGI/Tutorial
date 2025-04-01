import lazyllm

test1 = lambda a: a + 1
test2 = lambda a: a * 4
test3 = lambda a: a / 2

prl1 = lazyllm.parallel(test1, test2, test3)
prl2 = lazyllm.parallel(path1=test1, path2=test2, path3=test3).asdict
prl3 = lazyllm.parallel(test1, test2, test3).astuple
prl4 = lazyllm.parallel(test1, test2, test3).aslist
prl5 = lazyllm.parallel(test1, test2, test3).join('，')

print("默认输出：prl1(1) -> ", prl1(1), type(prl1(1)))
print("输出字典：prl2(1) -> ", prl2(1), type(prl2(1)))
print("输出元组：prl3(1) -> ", prl3(1), type(prl3(1)))
print("输出列表：prl4(1) -> ", prl4(1), type(prl4(1)))
print("输出字符串：prl5(1) -> ", prl5(1), type(prl5(1)))