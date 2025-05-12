import lazyllm
from lazyllm import ThreadPoolExecutor
from lazyllm import globals, FileSystemQueue
from pydantic import BaseModel
from typing import List, Optional
from lazyllm import LOG
import contextvars
from typing import Any, Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor

llm = lazyllm.OnlineChatModule(stream=True)
threadPool = ThreadPoolExecutor(max_workers=50)
slots = [0] * 50

DEFAULT_FEW_SHOTS = []


class ChatHistory(BaseModel):
    user: str
    assistant: str

class ChatRequest(BaseModel):
    user_input: str
    history: Optional[List[ChatHistory]] = None

def allocate_slot():
    for idx, val in enumerate(slots):
        if val == 0:
            slots[idx] = 1
            return idx
    return -1

def release_slot(session_id):
    if 0 <= session_id < len(slots):
        slots[session_id] = 0
        globals.pop(session_id)
    
llm = lazyllm.OnlineChatModule(stream=True)


def init_session_config(session_id, user_history=None):
    globals._init_sid(session_id)
    # if globals._sid not in globals._Globals__data:
    #     globals._Globals__data[globals._sid] = copy.deepcopy(globals.__global_attrs__)

    if user_history is not None:
        globals["global_parameters"]["history"] = user_history
    else:
        if "history" not in globals["global_parameters"]:
            globals["global_parameters"]["history"] = []


def with_session(func):
    """自动绑定 session_id 的装饰器"""
    def wrapper(session_id, *args, **kwargs):
        globals._init_sid(session_id)
        return func(session_id, *args, **kwargs)
    return wrapper


class SessionResponder:
    def __init__(self):
        pass

    def respond_stream(self, session_id, model_in, user_history=None):
        init_session_config(session_id, user_history)

        print("[Respond Stream] Current SID:", globals._sid)
        history = globals["global_parameters"]["history"]
        print("history", history)

        # 捕获当前上下文（确保线程池提交的任务也带上下文）
        ctx = contextvars.copy_context()
        func_future = threadPool.submit(lambda: ctx.run(llm, model_in, llm_chat_history=history))

        response = ''

        while True:
            assert session_id == globals._sid, f"\nSession ID mismatch: expected {session_id}, got {globals._sid}"

            if message := FileSystemQueue().dequeue():
                msg = "".join(message)
                response += msg
                yield msg
            elif func_future.done():
                break

        model_out = func_future.result()

        assert session_id == globals._sid, f"Session ID mismatch after LLM: expected {session_id}, got {globals._sid}"

        # globals["global_parameters"]["history"].append(model_out)
        globals["global_parameters"]["history"].append({
            "role": "user",
            "content": model_in
        })
        globals["global_parameters"]["history"].append({
            "role": "assistant",
            "content": model_out
        })

        return model_out


# 外部使用示例
@with_session
def handle_request(session_id: str, user_input: str):
    chat = SessionResponder()
    for chunk in chat.respond_stream(session_id, model_in=user_input):
        print(chunk, end='', flush=True)


if __name__ == "__main__":
    
    handle_request("user321", "苹果的英文是什么！")
    print("\n\n")
    handle_request("user123", "机器学习是什么")
    print("\n\n")
    handle_request("user321", "香蕉呢")
    print("\n\n")
    handle_request("user123", "它有什么用？")