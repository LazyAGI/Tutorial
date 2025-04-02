import time
from lazyllm.tools import Document

path = "path/to/docs"
docs = Document(path, manager='ui')  # manager=True 时启动后端服务
# 注册分组
Document(path, name='法务文档管理组', manager=docs.manager)
Document(path, name='产品文档管理组', manager=docs.manager)
# 启动服务
docs.start()
time.sleep(3600)