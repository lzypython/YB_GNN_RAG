import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from train_gnn import GCN  # 导入 GCN 模型结构

GRAPH_PATH = "graph.pkl"
MODEL_PATH = "gnn_model.pt"
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
    # print("候选实体及其得分：", [(n, float(scores[i])) for i, n in enumerate(ranked_nodes)])
    # 使用 LLM 生成最终答案
    context = "\n".join([f"{n}: {G.nodes[n]['text']}" for n in ranked_nodes[:topk]])
    prompt = f"""
    你是一位医学专家。
    问题：{query}
    相关候选实体及上下文：
    {context}
    进行详细充分的回答。
    """
    resp = client.chat.completions.create(
        model="qwen2.5-7b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    answer = resp.choices[0].message.content.strip()
    return answer

if __name__ == "__main__":
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # 获取节点 embedding 维度
    node_emb_dim = len(G.nodes[list(G.nodes())[0]]['embedding'])

    # 定义模型结构并加载参数
    gnn_model = GCN(in_dim=node_emb_dim, hidden_dim=64)
    gnn_model.load_state_dict(torch.load(MODEL_PATH))

    query = "医疗保险是什么？"
    ans = rag_query(query, G, gnn_model, topk=5)
    print(f"问：{query}\n答：{ans}")
