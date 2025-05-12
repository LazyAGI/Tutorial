import lazyllm
from lazyllm import ThreadPoolExecutor
from lazyllm import globals, FileSystemQueue
from pydantic import BaseModel
from typing import List, Optional
import contextvars
from concurrent.futures import ThreadPoolExecutor
import copy

llm = lazyllm.OnlineChatModule(stream=True)
threadPool = ThreadPoolExecutor(max_workers=50)
slots = [0] * 50

# 公共 few shot 历史
DEFAULT_FEW_SHOTS = [
    {"role": "user", "content": "你是谁？"},
    {"role": "assistant", "content": "我是你的智能助手。"}
]

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

def init_session_config(session_id, user_history=None):
    globals._init_sid(session_id)

    if user_history is not None:
        history = []
        # 合并 few-shot + 用户历史
        history.extend(DEFAULT_FEW_SHOTS)
        for h in user_history:
            history.append({"role": "user", "content": h.user})
            history.append({"role": "assistant", "content": h.assistant})
        globals["global_parameters"]["history"] = history
    else:
        if "history" not in globals["global_parameters"]:
            globals["global_parameters"]["history"] = copy.deepcopy(DEFAULT_FEW_SHOTS)

def with_session(func):
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

        # 更新历史
        globals["global_parameters"]["history"].append({
            "role": "user",
            "content": model_in
        })
        globals["global_parameters"]["history"].append({
            "role": "assistant",
            "content": model_out
        })

        return model_out

@with_session
def handle_request(session_id: str, user_input: str, user_history: Optional[List[ChatHistory]] = None):
    chat = SessionResponder()
    for chunk in chat.respond_stream(session_id, model_in=user_input, user_history=user_history):
        print(chunk, end='', flush=True)


if __name__ == "__main__":

    # 指定有历史的请求
    history = [
        ChatHistory(user="你好", assistant="你好呀！"),
        ChatHistory(user="你能帮我翻译成英文吗？", assistant="当然可以，请告诉我你需要翻译的内容。")
    ]
    handle_request("user123", "香蕉", user_history=history)
    print("\n\n")

    handle_request("user123", "总结这段对话")
