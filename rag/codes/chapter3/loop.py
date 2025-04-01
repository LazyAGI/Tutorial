import lazyllm

# 条件函数
stop_func = lambda x: x > 10

# 分支函数
module_func = lambda x: x * 2

# loop构建1
loop1 = lazyllm.loop(
    module_func,
    stop_condition=stop_func)

# Show:
print('1输出：', loop1(1))

#==========================
# 分支函数2
def module_func2(x):
    print("\tloop: ", x)
    return lazyllm.package(x+1, x*2)

# loop构建2
loop2 = lazyllm.loop(
    module_func2,
    stop_condition=stop_func,
    judge_on_full_input=False)

# Show:
print('2输出：', loop2(1))