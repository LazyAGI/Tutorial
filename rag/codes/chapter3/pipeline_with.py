import lazyllm

f1 = lambda x: x * 2

def f2(input):
  return input - 1

class AddOneFunctor(object):
  def __call__(self, x): return x + 1

f3 = AddOneFunctor()

# 使用with方式的数据流
with lazyllm.pipeline() as ppl:
    ppl.func1 = f1
    ppl.func2 = f2
    ppl.func3 = f3
    ppl.func4 = AddOneFunctor

inp = 2
out_ppl1 = ppl(inp)

print(f"输入为{inp},数据流输出:", out_ppl1)