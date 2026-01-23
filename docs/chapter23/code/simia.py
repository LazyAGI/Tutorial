import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class FeedbackType(Enum):
    SUCCESS = "success"
    ERROR = "error" 
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    RESOURCE_LOCKED = "resource_locked"

@dataclass
class SimulatedFeedback:
    """模拟环境反馈数据结构"""
    feedback_type: FeedbackType
    error_code: Optional[str] = None
    message: str = ""
    hidden_state_hint: Optional[str] = None
    observation: Optional[str] = None
    retry_suggestion: Optional[str] = None
    delay_seconds: int = 0

class SimiaSimulator:
    """Simia框架的核心：模拟器LLM"""
    
    def __init__(self, environment_description: str):
        self.environment_description = environment_description
        self.error_injection_rate = 0.1  # 10%的概率注入错误
        
    def simulate_feedback(self, agent_action: str, step: int) -> SimulatedFeedback:
        """
        根据Agent的动作生成模拟反馈
        这是Simia框架的核心函数
        """
        
        # 错误注入：模拟真实世界的不可预测性
        if random.random() < self.error_injection_rate:
            return self._generate_error_feedback(agent_action, step)
        
        # 正常反馈：基于环境描述和动作生成合理响应
        return self._generate_success_feedback(agent_action, step)
    
    def _generate_error_feedback(self, agent_action: str, step: int) -> SimulatedFeedback:
        """生成错误反馈，用于训练鲁棒性"""
        error_types = [
            FeedbackType.TIMEOUT,
            FeedbackType.PERMISSION_DENIED, 
            FeedbackType.RESOURCE_LOCKED
        ]
        
        error_type = random.choice(error_types)
        
        error_configs = {
            FeedbackType.TIMEOUT: {
                "error_code": "CONNECTION_TIMEOUT",
                "message": "Connection timed out after 30 seconds",
                "retry_suggestion": "Implement exponential backoff: wait 2^attempt seconds",
                "delay_seconds": 30
            },
            FeedbackType.PERMISSION_DENIED: {
                "error_code": "ACCESS_DENIED", 
                "message": "Permission denied: insufficient privileges",
                "retry_suggestion": "Check authentication credentials or request elevated permissions",
                "delay_seconds": 0
            },
            FeedbackType.RESOURCE_LOCKED: {
                "error_code": "RESOURCE_BUSY",
                "message": "Resource temporarily locked by another process",
                "retry_suggestion": "Wait for resource release or use optimistic locking",
                "delay_seconds": 5
            }
        }
        
        config = error_configs[error_type]
        
        return SimulatedFeedback(
            feedback_type=error_type,
            error_code=config["error_code"],
            message=config["message"],
            retry_suggestion=config["retry_suggestion"],
            delay_seconds=config["delay_seconds"],
            hidden_state_hint=f"Step {step}: Agent attempted '{agent_action}' but encountered {error_type.value}"
        )
    
    def _generate_success_feedback(self, agent_action: str, step: int) -> SimulatedFeedback:
        """生成成功反馈"""
        
        # 基于环境描述和动作生成合理的观察结果
        if "database" in self.environment_description.lower():
            # 数据库环境模拟
            if "SELECT" in agent_action.upper():
                observation = f"Query executed successfully. Returned 42 rows from products table."
            elif "INSERT" in agent_action.upper():
                observation = f"Record inserted successfully. New ID: {random.randint(1000, 9999)}"
            elif "UPDATE" in agent_action.upper():
                observation = f"Updated {random.randint(1, 10)} rows successfully."
            else:
                observation = "Command executed without errors."
                
        elif "web" in self.environment_description.lower():
            # Web服务环境模拟
            if "GET" in agent_action.upper():
                observation = "HTTP 200 OK. Response received with JSON payload."
            elif "POST" in agent_action.upper():
                observation = "HTTP 201 Created. Resource created successfully."
            else:
                observation = "Request processed successfully."
        else:
            observation = f"Action '{agent_action}' completed successfully in {self.environment_description}"
        
        return SimulatedFeedback(
            feedback_type=FeedbackType.SUCCESS,
            observation=observation,
            hidden_state_hint=f"Step {step}: Successful execution of '{agent_action}'"
        )

class SimiaTrajectoryGenerator:
    """生成完整的模拟轨迹"""
    
    def __init__(self, simulator: SimiaSimulator):
        self.simulator = simulator
        self.max_steps = 10
        
    def generate_trajectory(self, initial_state: str, task_description: str) -> List[Dict]:
        """
        生成完整的Agent-环境交互轨迹
        这是Simia-SFT的核心数据生成过程
        """
        trajectory = []
        current_state = initial_state
        
        for step in range(self.max_steps):
            # Agent思考并生成动作（这里用规则模拟，实际中用LLM）
            agent_action = self._mock_agent_action(current_state, task_description, step)
            
            # 模拟器生成反馈
            feedback = self.simulator.simulate_feedback(agent_action, step)
            
            # 记录轨迹
            step_data = {
                "step": step,
                "agent_action": agent_action,
                "environment_feedback": {
                    "feedback_type": feedback.feedback_type.value,
                    "error_code": feedback.error_code,
                    "message": feedback.message,
                    "observation": feedback.observation,
                    "retry_suggestion": feedback.retry_suggestion,
                    "delay_seconds": feedback.delay_seconds
                },
                "hidden_state_hint": feedback.hidden_state_hint
            }
            
            trajectory.append(step_data)
            
            # 更新状态
            if feedback.feedback_type == FeedbackType.SUCCESS:
                current_state = feedback.observation or current_state
            else:
                # 错误情况下，Agent需要处理错误
                if feedback.retry_suggestion:
                    current_state = f"Error encountered: {feedback.message}. Suggestion: {feedback.retry_suggestion}"
                else:
                    break  # 不可恢复错误，结束轨迹
        
        return trajectory
    
    def _mock_agent_action(self, current_state: str, task_description: str, step: int) -> str:
        """模拟Agent生成动作（实际中用LLM）"""
        
        # 简单的规则-based动作生成，实际应该用LLM
        if "database" in task_description.lower():
            actions = [
                'QUERY "SELECT * FROM products WHERE price > 100"',
                'INSERT INTO products (name, price) VALUES ("New Product", 99.99)',
                'UPDATE products SET price = price * 0.9 WHERE category = "electronics"',
                'DELETE FROM products WHERE id = 101'
            ]
        elif "web" in task_description.lower():
            actions = [
                'GET /api/products',
                'POST /api/products {"name": "New Item", "price": 29.99}',
                'PUT /api/products/123 {"price": 39.99}',
                'DELETE /api/products/456'
            ]
        else:
            actions = [f"Action_{step}", f"Command_{step}"]
        
        return random.choice(actions)

# === 使用示例 ===

def demonstrate_simia_framework():
    """展示Simia框架的使用"""
    
    # 1. 定义环境
    environment_desc = "你是一个电商网站的后端数据库，包含products表"
    simulator = SimiaSimulator(environment_desc)
    
    # 2. 创建轨迹生成器
    trajectory_gen = SimiaTrajectoryGenerator(simulator)
    
    # 3. 生成训练数据
    initial_state = "Database connection established. Ready to execute queries."
    task = "管理电商产品数据：查询、添加、更新产品信息"
    
    trajectory = trajectory_gen.generate_trajectory(initial_state, task)
    
    # 4. 格式化输出
    print("### Simia框架生成的数据样本 ###\n")
    print(f"环境描述: {environment_desc}")
    print(f"任务: {task}\n")
    
    for step_data in trajectory:
        print(f"--- 步骤 {step_data['step']} ---")
        print(f"Agent动作: {step_data['agent_action']}")
        
        feedback = step_data['environment_feedback']
        print(f"反馈类型: {feedback['feedback_type']}")
        
        if feedback['error_code']:
            print(f"错误代码: {feedback['error_code']}")
            print(f"错误信息: {feedback['message']}")
            if feedback['retry_suggestion']:
                print(f"重试建议: {feedback['retry_suggestion']}")
        else:
            print(f"观察结果: {feedback['observation']}")
            
        print(f"延迟: {feedback['delay_seconds']}秒")
        print(f"隐藏状态提示: {step_data['hidden_state_hint']}\n")

if __name__ == "__main__":
    demonstrate_simia_framework()