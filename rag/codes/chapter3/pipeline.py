import lazyllm

f1 = lambda x: x * 2

def f2(input):
  return input - 1

class AddOneFunctor(object):
  def __call__(self, x): return x + 1

f3 = AddOneFunctor()

# 手动调用：
inp = 2
x1 = f1(inp)
x2 = f2(x1)
x3 = f3(x2)
out_normal = AddOneFunctor()(x3)

# 使用数据流
ppl = lazyllm.pipeline(f1, f2, f3, AddOneFunctor)
out_ppl1 = ppl(inp)

print(f"输入为{inp},手动调用输出:", out_normal)
print(f"输入为{inp},数据流输出:", out_ppl1)
