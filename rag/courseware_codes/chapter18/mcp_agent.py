import lazyllm
from lazyllm import ReactAgent


mcp_client1 = lazyllm.tools.MCPClient(
    command_or_url="python",
    args=["-m", "mcp_simple_arxiv"],
)
mcp_client2 = lazyllm.tools.MCPClient(
    command_or_url="python",
    args=["-m", "mcp_server_calculator"],
)

llm = lazyllm.OnlineChatModule(stream=False)

paper_agent = ReactAgent(llm, mcp_client1.get_tools(), return_trace=True)
calculator_agent = ReactAgent(llm, mcp_client2.get_tools(), return_trace=True)

if __name__ == "__main__":
    lazyllm.WebModule(paper_agent, port=range(23468, 23470)).start().wait()
