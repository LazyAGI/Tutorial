import lazyllm

# 条件函数
is_positive = lambda x: x > 0
is_negative = lambda x: x < 0

# 每个条件函数对应一个分支函数：
positive_path = lambda x: 2 * x
negative_path = lambda x : -x
default_path = lambda x : '000'

# switch构建1（x同时作为条件函数和分支函数的输入）：
with lazyllm.switch() as sw1:
    sw1.case(is_positive, positive_path)
    sw1.case(is_negative, negative_path)
    sw1.case('default', default_path)

# Show:
print('\n输入x同时作为条件函数和分支函数的输入：')
print("1Path Positive: ", sw1(2)) # 2不仅传入条件函数，2也传入对应的分支函数；
print("1Path Default:  ", sw1(0))
print("1Path Negative: ", sw1(-5))

# switch构建2（条件函数和分支函数的输入指定不同的值）：
with lazyllm.switch(judge_on_full_input=False) as sw2:  # 分离条件函数和分支函数的关键开关。注意默认是True
    sw2.case(is_positive, positive_path)
    sw2.case(is_negative, negative_path)
    sw2.case('default', default_path)

# Show:
print('\n输入x,y按位置分别作为条件函数和分支函数的输入：')
print("2Path Positive: ", sw2(-1,2)) # -1传入条件函数，2传入对应分支函数；
print("2Path Default:  ", sw2(1,2))
print("2Path Negative: ", sw2(0, 2))
