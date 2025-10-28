import json
import time
import openai
from typing import List, Dict

# 初始化客户端
client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6082/v1"
)

def read_knowledge_file(file_path: str) -> str:
    """读取知识文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
        return ""
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return ""

def generate_qa_pairs(knowledge_text: str, num_pairs: int = 100) -> List[Dict]:
    """生成药物医保准入评估的是非型问答对"""
    
    prompt = f"""
    请基于以下关于新增药物医保准入评估的语料内容，生成{num_pairs}个高质量的问题-答案对。

    要求：
    1. 所有问题都必须是以下格式：
       "Q:现在有一款药，[药物描述]，它是否合适纳入医保药物名录？"
       "A:[适合/不适合]纳入，[详细理由]"

    2. 药物描述应该基于语料内容，包括但不限于：
       - 药物类型和作用机制（如：血管紧张素受体脑啡肽酶抑制剂）
       - 治疗效果和临床数据
       - 目标患者人群
       - 价格和成本效益
       - 安全性特征
       - 与现有药物的比较

    3. 答案必须明确给出"适合纳入"或"不适合纳入"的判断，并提供基于语料的详细理由

    4. 需要生成两种类型的案例：
       - 适合纳入的正面案例（约60%）
       - 不适合纳入的负面案例（约40%）

    5. 负面案例的理由可以包括：
       - 价格过高，成本效益不佳
       - 目标人群过小
       - 安全性问题
       - 与现有药物相比优势不明显
       - 基金影响过大

    语料内容：
    {knowledge_text}

    请以JSON格式返回结果，格式如下：
    [
    {{
        "question": "Q:现在有一款药，是一种血管紧张素受体脑啡肽酶抑制剂，能够显著降低心衰患者住院风险，但价格较高，它是否合适纳入医保药物名录？",
        "answer": "A:适合纳入，虽然价格较高，但能显著降低心衰患者住院风险，从长期看可以减少整体医疗支出，具有较好的成本效益。",
        "judgment": "适合",
        "reasoning": "降低住院风险带来的长期效益超过药品价格增加"
    }},
    {{
        "question": "Q:现在有一款药，治疗效果与现有药物相当但价格是现有药物的三倍，它是否合适纳入医保药物名录？", 
        "answer": "A:不适合纳入，与现有药物疗效相当但价格过高，不符合成本效益原则，可能给医保基金带来不必要的负担。",
        "judgment": "不适合",
        "reasoning": "疗效无优势但价格显著更高"
    }},
    ...
    ]

    现在请生成{num_pairs}个药物医保准入评估的是非型问答对：
    """
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[
                {"role": "system", "content": "你是一个专业的医保药物评估专家。请基于提供的语料生成关于药物是否适合纳入医保的是非型问答对。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=10000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 尝试解析JSON响应
        try:
            qa_pairs = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                qa_pairs = json.loads(json_match.group())
            else:
                print("无法解析模型响应为JSON")
                return []
        
        # 验证和标准化问答对格式
        validated_pairs = []
        for pair in qa_pairs:
            if (isinstance(pair, dict) and 
                'question' in pair and 
                'answer' in pair):
                
                # 确保问题格式正确
                question = pair['question'].strip()
                if not question.startswith('Q:'):
                    question = 'Q:' + question
                if not question.endswith('它是否合适纳入医保药物名录？'):
                    question = question + '它是否合适纳入医保药物名录？'
                
                # 确保答案格式正确
                answer = pair['answer'].strip()
                if not answer.startswith('A:'):
                    answer = 'A:' + answer
                
                validated_pairs.append({
                    'question': question,
                    'answer': answer,
                    'judgment': pair.get('judgment', ''),
                    'reasoning': pair.get('reasoning', '')
                })
        
        # 统计正负样本比例
        suitable_count = sum(1 for pair in validated_pairs if '适合' in pair['answer'])
        unsuitable_count = len(validated_pairs) - suitable_count
        
        print(f"成功生成 {len(validated_pairs)} 个问答对")
        print(f"适合纳入: {suitable_count} 个, 不适合纳入: {unsuitable_count} 个")
        
        return validated_pairs
        
    except Exception as e:
        print(f"调用大模型时出错：{e}")
        return []
def generate_qa_pairs_cx(knowledge_text: str, num_pairs: int = 100) -> List[Dict]:
    """生成长护险失能风险预测的是非型问答对"""
    
    prompt = f"""
    请基于以下关于长护险失能风险预测的语料内容，生成{num_pairs}个高质量的问题-答案对。

    要求：
    1. 所有问题都必须是以下格式：
       "Q:现在有一位参保人，[参保人描述]，该参保人是否属于长护险失能高风险人群？"
       "A:[是/否]属于高风险人群，[详细理由]"

    2. 参保人描述应该基于语料内容，包括但不限于：
       - 年龄和性别
       - 基础疾病情况（如：脑卒中、帕金森病、关节炎等）
       - 功能障碍表现（如：肢体活动受限、认知障碍等）
       - 跌倒史和住院记录
       - 社会支持状况
       - 日常生活能力评分

    3. 答案必须明确给出"是"或"否"的判断，并提供基于语料的详细理由

    4. 需要生成两种类型的案例：
       - 高风险人群的正面案例（约60%）
       - 非高风险人群的负面案例（约40%）

    5. 高风险人群的判断标准包括：
       - 年龄大于75岁
       - 合并3种以上慢性病
       - 近一年有跌倒史
       - 认知功能下降（MoCA评分低）
       - 社会支持系统薄弱
       - Barthel指数低于60分

    6. 非高风险人群的理由可以包括：
       - 年龄较轻（小于65岁）
       - 慢性病数量少且控制良好
       - 无跌倒史或功能障碍
       - 认知功能正常
       - 社会支持系统完善

    语料内容：
    {knowledge_text}

    请以JSON格式返回结果，格式如下：
    [
    {{
        "question": "Q:现在有一位参保人，82岁男性，患有高血压和糖尿病20年，三年前脑梗死后遗留左侧肢体偏瘫，近两年因跌倒骨折住院3次，该参保人是否属于长护险失能高风险人群？",
        "answer": "A:是属于高风险人群，该参保人年龄超过80岁，合并多种慢性疾病，有脑卒中病史导致肢体功能障碍，近两年多次跌倒骨折，符合多个高风险特征，未来一年失能发生概率超过70%。",
        "judgment": "是",
        "reasoning": "高龄、多慢性病、功能障碍史、多次跌倒"
    }},
    {{
        "question": "Q:现在有一位参保人，65岁女性，仅有轻度高血压且控制良好，无跌倒史，日常生活完全自理，认知功能正常，该参保人是否属于长护险失能高风险人群？", 
        "answer": "A:不属于高风险人群，该参保人年龄相对较轻，慢性病单一且控制良好，无功能障碍和跌倒史，认知功能正常，不具备失能高风险特征。",
        "judgment": "否",
        "reasoning": "年龄较轻、慢性病单一控制好、无功能障碍"
    }},
    {{
        "question": "Q:现在有一位参保人，78岁独居女性，患有骨质疏松和重度膝关节炎，过去一年有两次跌倒记录，MoCA认知评估得分18分存在轻度认知障碍，该参保人是否属于长护险失能高风险人群？",
        "answer": "A:是属于高风险人群，该参保人高龄独居，患有关节炎影响活动能力，有跌倒史且存在认知障碍，六个月内发展为失能的概率高达65%，属于典型的高风险人群。",
        "judgment": "是", 
        "reasoning": "高龄独居、关节炎功能障碍、跌倒史、认知障碍"
    }},
    ...
    ]

    现在请生成{num_pairs}个长护险失能风险预测的是非型问答对：
    """
    
    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[
                {"role": "system", "content": "你是一个专业的长护险风险评估专家。请基于提供的语料生成关于参保人是否属于失能高风险人群的是非型问答对。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=10000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 尝试解析JSON响应
        try:
            qa_pairs = json.loads(result_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                qa_pairs = json.loads(json_match.group())
            else:
                print("无法解析模型响应为JSON")
                return []
        
        # 验证和标准化问答对格式
        validated_pairs = []
        for pair in qa_pairs:
            if (isinstance(pair, dict) and 
                'question' in pair and 
                'answer' in pair):
                
                # 确保问题格式正确
                question = pair['question'].strip()
                if not question.startswith('Q:'):
                    question = 'Q:' + question
                if not question.endswith('该参保人是否属于长护险失能高风险人群？'):
                    question = question + '该参保人是否属于长护险失能高风险人群？'
                
                # 确保答案格式正确
                answer = pair['answer'].strip()
                if not answer.startswith('A:'):
                    answer = 'A:' + answer
                
                validated_pairs.append({
                    'question': question,
                    'answer': answer,
                    'judgment': pair.get('judgment', ''),
                    'reasoning': pair.get('reasoning', ''),
                    'risk_factors': pair.get('risk_factors', '')  # 可选：记录具体风险因素
                })
        
        # 统计正负样本比例
        high_risk_count = sum(1 for pair in validated_pairs if '是' in pair['answer'] and '不属于' not in pair['answer'])
        low_risk_count = len(validated_pairs) - high_risk_count
        
        print(f"成功生成 {len(validated_pairs)} 个问答对")
        print(f"高风险人群: {high_risk_count} 个, 低风险人群: {low_risk_count} 个")
        
        return validated_pairs
        
    except Exception as e:
        print(f"调用大模型时出错：{e}")
        return []

def save_to_json(qa_pairs: List[Dict], output_file: str):
    """保存问答对到JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
        print(f"成功保存 {len(qa_pairs)} 个问答对到 {output_file}")
    except Exception as e:
        print(f"保存文件时出错：{e}")

def main():
    # 文件路径
    knowledge_file = "knowledge_cx.txt"
    output_file = "medical_insurance_qa_cx.json"

    # 读取语料
    print("正在读取语料文件...")
    knowledge_text = read_knowledge_file(knowledge_file)
    if not knowledge_text:
        print("语料文件为空或读取失败，程序退出")
        return
    
    print(f"语料文件读取成功，内容长度：{len(knowledge_text)} 字符")
    
    # 生成问答对
    print("正在生成问答对...")
    qa_pairs = generate_qa_pairs_cx(knowledge_text, 100)
    
    if not qa_pairs:
        print("生成问答对失败")
        return
    
    print(f"成功生成 {len(qa_pairs)} 个问答对")
    
    # 保存结果
    save_to_json(qa_pairs, output_file)
    
    # 显示前几个示例
    print("\n前5个问答对示例：")
    for i, pair in enumerate(qa_pairs[:5]):
        print(f"{i+1}. 问题：{pair['question']}")
        print(f"   答案：{pair['answer']}")
        print()

if __name__ == "__main__":
    main()