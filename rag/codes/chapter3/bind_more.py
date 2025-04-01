import lazyllm
from lazyllm import pipeline, parallel, bind, _0, _1

def f1(input): return input ** 2
def f21(input1, input2=0): return input1 + input2 + 1
def f22(input1, input2=0): return input1 + input2 - 1
def f3(in1='placeholder1', in2='placeholder2', in3='placeholder3'): 
    return f'get [input:{in1}], [f21:{in2}], [f23: {in3}]]'

with pipeline() as ppl1:
  ppl1.f1 = f1
  with parallel().bind(ppl1.input, _0) as ppl1.subprl2:
    ppl1.subprl2.path1 = f21
    ppl1.subprl2.path2 = f22
  ppl1.f3 = f3 | bind(ppl1.input, _0, _1)
  
print("ppl1 out: ", ppl1(2))