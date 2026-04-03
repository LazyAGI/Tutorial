import lazyllm
from lazyllm.tools.agent import ReactAgent
from lazyllm.tools import MCPClient
  
mcp_configs = {        
    "file_system": {            
        "command": "cmd",            
        "args": [                
            "/c",                
            "npx",                
            "-y",                
            "@modelcontextprotocol/server-filesystem",                
            "./"            
        ]        
    },        
    "play_wright": {            
        "url": "http://127.0.0.1:11244/sse"        
    }    
}    
client1 = MCPClient(command_or_url=mcp_configs["file_system"]["command"], args=mcp_configs["file_system"]["args"])    
client2 = MCPClient(command_or_url=mcp_configs["play_wright"]["url"])    
llm = lazyllm.OnlineChatModule()    
agent = ReactAgent(llm=llm.share(), tools=client1.get_tools()+client2.get_tools(), max_retries=15)    
print(agent("浏览谷歌新闻，并写一个今日新闻简报，以markdown格式保存至本地。"))
