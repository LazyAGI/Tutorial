import sympy as sp

def verify_math_equivalence(expr1_str, expr2_str):
    """
    验证两个数学表达式在符号层面是否等价
    例如: (x + 1)**2 和 x**2 + 2*x + 1
    """
    x = sp.symbols('x')
    
    try:
        # 将字符串转换为 SymPy 表达式
        expr1 = sp.simplify(expr1_str)
        expr2 = sp.simplify(expr2_str)
        
        # 方法：检查两者之差是否化简为 0
        diff = sp.simplify(expr1 - expr2)
        
        if diff == 0:
            return True, "验证通过：表达式逻辑一致"
        else:
            return False, f"验证失败：差值为 {diff}"
    except Exception as e:
        return False, f"解析错误: {str(e)}"

# 实践演示
steps = [
    ("(x + 1)**2", "x**2 + 2*x + 1"),   # 正确推导
    ("sin(x)**2 + cos(x)**2", "1"),     # 三角恒等式
    ("exp(x + y)", "exp(x) * exp(y)")   # 指数法则 (需定义y)
]

for e1, e2 in steps:
    is_valid, msg = verify_math_equivalence(e1, e2)
    print(f"验证 [{e1}] == [{e2}]: {is_valid}")