import traceback
import signal
import types

class SandboxTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    raise SandboxTimeout("Execution timed out")


def sandbox_run_code(code: str, test_fn=None, time_limit=2):
    """
    在受控沙箱中执行代码，并进行验证

    参数：
    - code: str，待执行的 Python 代码
    - test_fn: callable，可选，用于验证逻辑正确性的测试函数
    - time_limit: int，最大执行时间（秒）

    返回：
    - dict，包含执行状态与结果信息
    """

    # 限制可用内建函数（最小权限）
    safe_builtins = {
        "range": range,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
    }

    global_env = {
        "__builtins__": safe_builtins
    }
    local_env = {}

    # 设置超时信号
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(time_limit)

    try:
        # 执行用户代码
        exec(code, global_env, local_env)

        # 可选：执行验证函数
        if test_fn is not None:
            test_fn(local_env)

        signal.alarm(0)  # 取消超时

        return {
            "status": "pass",
            "message": "Code executed successfully",
            "env_keys": list(local_env.keys())
        }

    except SandboxTimeout as e:
        return {
            "status": "fail",
            "error_type": "timeout",
            "error": str(e)
        }

    except Exception as e:
        return {
            "status": "fail",
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    finally:
        signal.alarm(0)


# =========================
# 示例：沙箱验证一段函数代码
# =========================



code_syntax_error = """
def bad_func(a, b)
    return a + b
"""

print("Case 1: Syntax Error")
print(sandbox_run_code(code_syntax_error))


code_runtime_error = """
def divide(a, b):
    return a / b
"""

def test_divide(env):
    env["divide"](10, 0)

print("\nCase 2: Runtime Error (ZeroDivisionError)")
print(sandbox_run_code(code_runtime_error, test_fn=test_divide))


code_success = """
def sum_first_n(n):
    s = 0
    for i in range(1, n + 1):
        s += i
    return s
"""

def test_sum_first_n(env):
    assert "sum_first_n" in env
    assert env == 1
    assert env == 6
    assert env == 55

print("\nCase 3: Success Case")
print(sandbox_run_code(code_success, test_fn=test_sum_first_n))