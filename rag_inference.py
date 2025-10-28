import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from train_gnn import GCN  # 导入 GCN 模型结构

GRAPH_PATH = "graph_yw.pkl"
MODEL_PATH = "gnn_model_yw.pt"
EMBED_MODEL = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6081/v1"
)

def rag_query(query, G, gnn_model, topk=5, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentenceTransformer(EMBED_MODEL)
    query_emb = model.encode(query)

    # 对所有节点创建高效 tensor
    nodes = list(G.nodes())
    node_embs = np.stack([G.nodes[n]['embedding'] for n in nodes])
    node_embs = torch.tensor(node_embs, dtype=torch.float, device=device)

    # 计算余弦相似度
    query_emb_tensor = torch.tensor(query_emb, dtype=torch.float, device=device)
    sims = torch.cosine_similarity(query_emb_tensor.unsqueeze(0), node_embs)

    # topk 候选节点索引
    candidate_idx = sims.topk(topk).indices
    X = node_embs[candidate_idx]

    # 将 GNN 模型移到 device 并 eval
    gnn_model.to(device)
    gnn_model.eval()

    with torch.no_grad():
        edge_index = torch.arange(X.size(0)).unsqueeze(0).repeat(2,1).to(device)
        out = gnn_model(X, edge_index)
    scores = torch.softmax(out[:,1], dim=0)
    ranked_nodes = [nodes[i] for i in candidate_idx[torch.argsort(scores, descending=True)]]
    
    # 使用 LLM 生成最终答案
    context = "\n".join([f"{n}: {G.nodes[n]['text']}" for n in ranked_nodes[:topk]])
    
    prompt = f"""
    # 角色定位
    你是一位资深医保药物经济学评估专家，专门负责新增药物医保准入评估工作。请基于提供的专业知识，以正式评估报告的形式回答问题。

    # 评估问题
    **待评估问题**：{query}

    # 相关证据材料
    {context}

    # 报告格式要求
    请按照以下结构化格式撰写《新增药物医保准入评估报告》：

    ## 一、评估摘要
    - **核心结论**：明确给出是否建议纳入的初步判断
    - **关键依据**：简要说明主要评估维度和关键发现

    ## 二、目标人群分析
    ### 2.1 适用人群特征
    - 目标患者群体的临床特征
    - 预估患者规模及分布
    - 现有治疗缺口分析

    ### 2.2 人群细分价值
    - 核心获益人群识别
    - 各亚组预期获益程度

    ## 三、基金影响评估
    ### 3.1 直接成本分析
    - 预期药品费用支出
    - 不同渗透率下的敏感性分析

    ### 3.2 间接效益评估
    - 预期医疗资源节约（住院、急诊等）
    - 长期成本效益分析

    ## 四、临床价值评估
    ### 4.1 疗效优势分析
    - 与现有标准治疗的比较
    - 临床终点改善情况

    ### 4.2 安全性评价
    - 不良反应风险评估
    - 特殊人群使用注意事项

    ## 五、综合建议
    ### 5.1 准入建议
    - 明确建议：□建议纳入 □有条件纳入 □不建议纳入
    - 建议支付标准及限制条件

    ### 5.2 风险管控措施
    - 建议的监测指标
    - 费用控制配套措施

    ## 六、证据等级说明
    - 数据来源及可靠性评估
    - 主要不确定性及局限性

    # 写作要求
    1. 采用专业、客观的评估语言
    2. 所有结论必须基于提供的证据材料
    3. 关键判断需注明数据支撑
    4. 体现医保基金可持续性考量
    5. 突出患者获益与基金平衡的权衡

    请开始撰写评估报告：
    """
    
    resp = client.chat.completions.create(
        model="qwen2.5-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = resp.choices[0].message.content.strip()
    answer = answer.replace("```", "").replace("json", "").strip()
    answer = answer.replace("#", "").strip()
    answer = answer.replace("*", "").strip()
    return answer
def rag_query_cx(query, G, gnn_model, topk=5, device=None):
    """长护险失能风险预测专家评估函数"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SentenceTransformer(EMBED_MODEL)
    query_emb = model.encode(query)

    # 对所有节点创建高效 tensor
    nodes = list(G.nodes())
    node_embs = np.stack([G.nodes[n]['embedding'] for n in nodes])
    node_embs = torch.tensor(node_embs, dtype=torch.float, device=device)

    # 计算余弦相似度
    query_emb_tensor = torch.tensor(query_emb, dtype=torch.float, device=device)
    sims = torch.cosine_similarity(query_emb_tensor.unsqueeze(0), node_embs)

    # topk 候选节点索引
    candidate_idx = sims.topk(topk).indices
    X = node_embs[candidate_idx]

    # 将 GNN 模型移到 device 并 eval
    gnn_model.to(device)
    gnn_model.eval()

    with torch.no_grad():
        edge_index = torch.arange(X.size(0)).unsqueeze(0).repeat(2,1).to(device)
        out = gnn_model(X, edge_index)
    scores = torch.softmax(out[:,1], dim=0)
    ranked_nodes = [nodes[i] for i in candidate_idx[torch.argsort(scores, descending=True)]]
    
    # 使用 LLM 生成最终答案
    context = "\n".join([f"{n}: {G.nodes[n]['text']}" for n in ranked_nodes[:topk]])
    
    prompt = f"""
    # 角色定位
    你是一位资深长期护理保险风险评估专家，专门负责参保人失能风险预测和护理需求评估工作。请基于提供的专业知识，以正式风险评估报告的形式回答问题。

    # 评估问题
    **待评估问题**：{query}

    # 相关证据材料
    {context}

    # 报告格式要求
    请按照以下结构化格式撰写《长护险失能风险评估报告》：

    ## 一、风险评估摘要
    - **核心结论**：明确给出风险等级判断（高风险/中风险/低风险）
    - **风险概率**：基于模型预测的失能发生概率估算
    - **关键风险因素**：识别主要风险驱动因素

    ## 二、参保人基本情况分析
    ### 2.1 人口学特征
    - 年龄、性别等基本信息分析
    - 社会支持系统评估（独居、家庭支持等）

    ### 2.2 健康状况评估
    - 基础疾病数量及严重程度
    - 慢性病控制情况分析
    - 既往重要病史（脑卒中、骨折等）

    ## 三、功能状态评估
    ### 3.1 日常生活能力
    - Barthel指数或ADL评分分析
    - 具体功能障碍表现（进食、洗澡、穿衣等）

    ### 3.2 认知与心理状态
    - 认知功能评估（MoCA评分等）
    - 精神状态及情绪评估

    ### 3.3 活动能力
    - 跌倒史及跌倒风险分析
    - 肢体活动受限程度
    - 平衡能力和行走能力

    ## 四、风险等级判定
    ### 4.1 风险因素汇总
    - 主要风险因素清单及权重
    - 风险因素交互影响分析

    ### 4.2 风险等级划分
    - □ 高风险（预测概率>60%）
    - □ 中风险（预测概率30%-60%）  
    - □ 低风险（预测概率<30%）

    ## 五、护理需求与干预建议
    ### 5.1 护理等级建议
    - 建议的长护险护理等级
    - 推荐服务频次和时长

    ### 5.2 早期干预措施
    - 预防性护理服务建议
    - 康复训练指导
    - 居家适老化改造建议

    ### 5.3 监测随访计划
    - 风险动态监测指标
    - 建议随访频率
    - 预警阈值设置

    ## 六、成本效益分析
    ### 6.1 预期护理成本
    - 不同护理等级下的费用估算
    - 长期护理费用趋势预测

    ### 6.2 干预效益评估
    - 早期干预的潜在费用节约
    - 生活质量改善预期

    ## 七、证据等级说明
    - 数据来源及完整性评估
    - 风险评估的不确定性说明
    - 建议补充的评估项目

    # 写作要求
    1. 采用专业、客观的临床评估语言
    2. 所有结论必须基于提供的证据材料
    3. 风险判断需明确标注数据支撑
    4. 体现预防为主、早期干预的理念
    5. 突出个性化、精准化的护理建议
    6. 考虑长护险基金可持续性

    请开始撰写风险评估报告：
    """
    
    resp = client.chat.completions.create(
        model="qwen2.5-32b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = resp.choices[0].message.content.strip()
    # 清理格式标记
    answer = answer.replace("```", "").replace("json", "").strip()
    answer = answer.replace("#", "").strip()
    answer = answer.replace("*", "").strip()
    return answer
if __name__ == "__main__":
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # 获取节点 embedding 维度
    node_emb_dim = len(G.nodes[list(G.nodes())[0]]['embedding'])

    # 定义模型结构并加载参数
    gnn_model = GCN(in_dim=node_emb_dim, hidden_dim=64)
    gnn_model.load_state_dict(torch.load(MODEL_PATH))

    query = "现在有一款药，是专门用于治疗射血分数降低的心力衰竭的新型心肌肌球蛋白激活剂，能够显著降低心血管死亡风险和心衰相关住院风险，它是否合适纳入医保药物名录？"
    query_cx = "现在有一位78岁独居女性，患有骨质疏松和重度膝关节炎，过去一年有两次跌倒记录，MoCA认知评估得分18分，请评估其失能风险等级并给出护理建议"
    ans = rag_query_cx(query_cx, G, gnn_model, topk=5)
    print(f"问：{query_cx}\n答：{ans}")
