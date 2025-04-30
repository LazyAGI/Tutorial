import lazyllm
from lazyllm.tools.eval import LLMContextRecall, NonLLMContextRecall, ContextRelevance

# 检索组件要求准备满足如下格式要求的数据进行评估
data = [{'question': '非洲的猴面包树果实的长度约是多少厘米？',
         # 当使用基于LLM的评估方法时要求答案是标注的正确答案
         'answer': '非洲猴面包树的果实长约15至20厘米。',
         # context_retrieved 为召回器召回的文档，按段落输入为列表
         'context_retrieved': ['非洲猴面包树是一种锦葵科猴面包树属的大型落叶乔木，原产于热带非洲，它的果实长约15至20厘米。',
                              '钙含量比菠菜高50％以上，含较高的抗氧化成分。',],
         # context_reference 为标注的应当被召回的段落
         'context_reference': ['非洲猴面包树是一种锦葵科猴面包树属的大型落叶乔木，原产于热带非洲，它的果实长约15至20厘米。']
}]
# 返回召回文档的命中率，例如上述data成功召回了标注的段落，因此召回率为1
m_recall = NonLLMContextRecall()
print(m_recall(data)) # 1.0

# 返回召回文档中的上下文相关性分数，例如上述data召回的两个句子中只有一个是相关的
m_cr = ContextRelevance()
print(m_cr(data)) # 0.5

# 返回基于LLM计算的召回率，LLM基于answer和context_retrieved判断是否召回了所有相关文档
# 适用于没有标注的情况下使用，比较耗费 token ，请根据需求谨慎使用
m_lcr = LLMContextRecall(lazyllm.OnlineChatModule(source='sensenova', model='SenseChat-5'))
print(m_lcr(data)) 