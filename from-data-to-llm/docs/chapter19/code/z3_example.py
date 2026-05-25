from z3 import Int, If, Solver, Not, unsat


def verify_abs_logic():
    '''
    使用 Z3 验证 abs(x) 的逻辑实现是否正确
    性质：对于任意整数 x，结果 res 必须满足 res >= 0
    '''
    # 1. 定义符号变量
    x = Int('x')
    res = Int('res')

    # 2. 待验证的逻辑实现 (这里是正确的逻辑)
    # 逻辑：res = (如果 x > 0 则为 x，否则为 -x)
    logic_implementation = (res == If(x > 0, x, -x))

    # 3. 明确需要验证的性质
    property_to_prove = (res >= 0)

    # 4. 核心：寻找反例 (Proof by Counterexample)
    # 原理：如果 (逻辑成立) 且 (性质不成立) 是不可满足的 (unsat)，则性质永远成立
    solver = Solver()
    solver.add(logic_implementation)
    solver.add(Not(property_to_prove))  # 试图寻找 res < 0 的情况

    print('--- 正在进行逻辑一致性验证 ---')
    check = solver.check()

    if check == unsat:
        return '✅ 验证通过：在给定约束范围内，逻辑具有完全一致性，未发现反例。'
    else:
        # 如果找到反例，提取具体的数值
        counterexample = solver.model()
        return (f'❌ 验证失败：找到逻辑漏洞！'
                f'当输入为 x = {counterexample[x]} 时，性质不成立。')


if __name__ == '__main__':
    result = verify_abs_logic()
    print(result)
