import lazyllm

# 条件函数
is_positive = lambda x: x > 0
is_negative = lambda x: x < 0

# 每个条件函数对应一个分支函数：
positive_path = lambda x: 2 * x
negative_path = lambda x : -x
default_path = lambda x : '000'

# switch构建1（x同时作为条件函数和分支函数的输入）：
switch1 = lazyllm.switch(
    is_positive, positive_path,
    is_negative, negative_path,
    'default', default_path)

# Show:
print('\n输入x同时作为条件函数和分支函数的输入：')
print("1Path Positive: ", switch1(2)) # 2不仅传入条件函数，2也传入对应的分支函数；
print("1Path Default:  ", switch1(0))
print("1Path Negative: ", switch1(-5))

# switch构建2（条件函数和分支函数的输入指定不同的值）：
switch2 = lazyllm.switch(
    is_positive, positive_path,
    is_negative, negative_path,
    'default', default_path,
    judge_on_full_input=False)

# Show:
print('\n输入x,y按位置分别作为条件函数和分支函数的输入：')
print("2Path Positive: ", switch2(-1,2)) # -1传入条件函数，2传入对应分支函数；
print("2Path Default:  ", switch2(1,2))
print("2Path Negative: ", switch2(0, 2))
