import lazyllm

coder = lazyllm.OnlineChatModule(source="sensenova", model="SenseChat-5")

def run_test(code, test_case):
    try:
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
            
        local_vars = {}
        exec(code + "\n" + test_case, {"__builtins__": __builtins__}, local_vars)
        return True
    except Exception as e:
        print(f"代码执行失败: {e}")
        return False

def construct_code_preference(requirement):

    prompt_a = f"Write a Python function for: {requirement}. Output only the code block."
    prompt_b = f"Write a Python function for: {requirement} using a different logical approach. Output only the code block."

    response_a = coder(prompt_a)
    response_b = coder(prompt_b)
    
    test_case = "assert solve([1, 2, 3]) == 6"
    
    passed_a = run_test(response_a, test_case)
    passed_b = run_test(response_b, test_case)
    
    print(f"Result A: {passed_a}, Result B: {passed_b}")

    if passed_a and not passed_b:
        return {"prompt": requirement, "chosen": response_a, "rejected": response_b}
    elif passed_b and not passed_a:
        return {"prompt": requirement, "chosen": response_b, "rejected": response_a}
    else:
        return "No clear winner (Both passed or both failed)"
运行
data = construct_code_preference("Write a function solve(arr) that calculates the sum of an array")
print("\n--- Final Data ---")
print(data)