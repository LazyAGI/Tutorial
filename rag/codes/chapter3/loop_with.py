import lazyllm

# 条件函数
stop_func = lambda x: x > 10

# 分支函数
module_func = lambda x: x
modele_func2 = lambda x: x * 2

# loop构建
with lazyllm.loop(stop_condition=stop_func) as loop1:
    loop1.func1 = module_func
    loop1.func2 = modele_func2

# Show:
print('输出：', loop1(1))

#==========================
# 分支函数2
def module_funcn2(x):
    print("\tloop: ", x)
    return lazyllm.package(x+1, x*2)

# loop构建2
with lazyllm.loop(stop_condition=stop_func, judge_on_full_input=False) as loop2:
    loop2.func1 = module_func
    loop2.func2 = module_funcn2

# Show:
print('2输出：', loop2(1))