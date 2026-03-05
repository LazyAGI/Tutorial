import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class MathProblem:
    """数学应用题数据结构"""
    question: str
    plan: List[str]
    execution: List[str]
    final_answer: Any

class PlanSolveAgent:
    """Plan & Solve Agent 实现"""
    
    def __init__(self):
        self.plan_templates = {
            "relationship": "根据{entity1}的数量，计算{entity2}的数量",
            "sum": "计算{entities}的总和",
            "multiply": "根据{base}计算{target}的数量"
        }
        
    def extract_entities(self, question: str) -> Dict[str, Any]:
        """从问题中提取实体和关系"""
        entities = {}
        
        # 提取人名和数量
        patterns = {
            "person_count": r"([小明|小红|小刚])有(\d+)个苹果",
            "relationship": r"([小红|小刚])的苹果是([小明|小红])的(\d+)倍(少|多)(\d+)个",
            "total_relationship": r"([小刚])的苹果是([小红|小明]+.*总和)的(\d+)倍"
        }
        
        # 解析基础数量
        for match in re.finditer(patterns["person_count"], question):
            person, count = match.groups()
            entities[person] = {"count": int(count), "relation": "given"}
        
        # 解析关系
        for match in re.finditer(r"([小红|小刚])的苹果是([小明|小红])的(\d+)倍(少|多)(\d+)个", question):
            target, base, multiplier, operation, offset = match.groups()
            entities[target] = {
                "relation": "calculated",
                "base": base,
                "multiplier": int(multiplier),
                "operation": operation,
                "offset": int(offset)
            }
        
        # 解析总和关系
        for match in re.finditer(r"([小刚])的苹果是([小红|小明]+.*总和)的(\d+)倍", question):
            target, base_desc, multiplier = match.groups()
            entities[target] = {
                "relation": "total_multiple",
                "base_description": base_desc,
                "multiplier": int(multiplier)
            }
        
        return entities
    
    def generate_plan(self, question: str) -> List[str]:
        """生成解题计划"""
        entities = self.extract_entities(question)
        
        plan = []
        step_num = 1
        
        # 找出需要计算的实体顺序
        calculation_order = []
        
        # 首先处理直接关系
        for name, info in entities.items():
            if info["relation"] == "calculated":
                calculation_order.append(name)
        
        # 然后处理基于总和的关系
        for name, info in entities.items():
            if info["relation"] == "total_multiple":
                calculation_order.append(name)
        
        # 生成计划步骤
        for entity in calculation_order:
            info = entities[entity]
            if info["relation"] == "calculated":
                base = info["base"]
                plan.append(f"{step_num}. 根据{base}的苹果数量，计算{entity}的苹果数量。")
            elif info["relation"] == "total_multiple":
                plan.append(f"{step_num}. 计算{entity}的苹果数量（基于总和的倍数关系）。")
            step_num += 1
        
        return plan
    
    def execute_plan(self, question: str, plan: List[str]) -> Dict[str, Any]:
        """执行解题计划"""
        entities = self.extract_entities(question)
        execution_steps = []
        calculations = {}
        
        # 初始化已知数量
        for name, info in entities.items():
            if info["relation"] == "given":
                calculations[name] = info["count"]
                execution_steps.append(f"**{name}**：{info['count']} 个（已知）")
        
        # 按计划执行计算
        step_num = 1
        for plan_step in plan:
            if "根据" in plan_step and "计算" in plan_step:
                # 解析是哪个实体的计算
                if "小红" in plan_step:
                    # 计算小红：小明的2倍少3个
                    xiaoming = calculations["小明"]
                    xiaohong = (xiaoming * 2) - 3
                    calculations["小红"] = xiaohong
                    execution_steps.append(f"{step_num}. **计算小红**：小明 {xiaoming} 个，({xiaoming} * 2) - 3 = {xiaohong} 个。")
                
                elif "小刚" in plan_step and "总和" in plan_step:
                    # 计算小明和小红的总和
                    total = calculations["小明"] + calculations["小红"]
                    calculations["总和"] = total
                    execution_steps.append(f"{step_num}. **计算总和**：{calculations['小明']} + {calculations['小红']} = {total} 个。")
                    
                    # 然后计算小刚：总和的3倍
                    step_num += 1
                    xiaogang = total * 3
                    calculations["小刚"] = xiaogang
                    execution_steps.append(f"{step_num}. **计算小刚**：{total} * 3 = {xiaogang} 个。")
            
            step_num += 1
        
        return {
            "calculations": calculations,
            "execution_steps": execution_steps,
            "final_answer": calculations.get("小刚", "未知")
        }

def demonstrate_plan_solve():
    """演示Plan & Solve方法"""
    
    # 问题定义
    question = "小明有10个苹果，小红的苹果是小明的2倍少3个，小刚的苹果是小红和小明总和的3倍，问小刚有多少苹果？"
    
    # 创建Agent
    agent = PlanSolveAgent()
    
    print("### 📝 复杂数学应用题 - Plan & Solve 演示")
    print(f"\n**用户问题**：{question}")
    
    # Plan阶段
    print("**Plan 阶段**:")
    plan = agent.generate_plan(question)
    print("**Plan**:")
    for step in plan:
        print(f"> {step}")
    
    # Solve阶段
    print("**Solve 阶段**:")
    result = agent.execute_plan(question, plan) 
    
    print("**Execution**:")
    for step in result["execution_steps"]:
        print(f"> {step}")
    
    print(f"\n**Final Answer**: {result['final_answer']}")
    
    # 显示详细计算过程
    print("**详细计算结果**:")
    for name, count in result["calculations"].items():
        print(f"- {name}: {count} 个")

if __name__ == "__main__":
    demonstrate_plan_solve()