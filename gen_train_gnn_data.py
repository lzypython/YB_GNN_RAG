import pickle
import networkx as nx
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import openai
import os

GRAPH_PATH = "graph_cx.pkl"
QUERIES_PATH = "gnn_queries_cx.pkl"
TRAIN_DATA_PATH = "gnn_train_data_cx.pt"
EMBED_MODEL = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6082/v1"
)

# PPR 获取候选实体
def get_ppr_topk(G, alpha=0.85, topk=5):
    ppr_scores = {}
    for node in G.nodes():
        scores = nx.pagerank(G, alpha=alpha, personalization={node: 1})
        top_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        ppr_scores[node] = [n for n, s in top_nodes]
    return ppr_scores

# 用 LLM 对候选实体进行二分类
def classify_candidates_with_llm(query, candidates, G):
    context = "\n".join([f"{c}: {G.nodes[c]['text']}" for c in candidates])
    prompt = f"""
    你是一位医学专家。
    问题：{query}
    以下候选实体及其上下文，请判断每个实体是否是问题的答案，输出格式：实体名称 | 0或1，1表示答案，0表示不是答案。
    {context}
    """
    try:
        resp = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        output = resp.choices[0].message.content
        labels = {}
        for line in output.split('\n'):
            if '|' in line:
                n, v = line.strip().split('|')
                n, v = n.strip(), int(v.strip())
                if n in candidates:
                    labels[n] = v
        for c in candidates:
            if c not in labels:
                labels[c] = 0
        return labels
    except Exception:
        return {c: 0 for c in candidates}

# 生成训练数据
def generate_training_data(G, queries, topk=5):
    model = SentenceTransformer(EMBED_MODEL)
    X, y = [], []

    for q in tqdm(queries, desc="生成训练样本"):
        query_text = q['question']
        answer_entity = q['answer']
        candidate_nodes = list(G.nodes())[:topk]  # 可以用 PPR 或其他方法获取 topk 候选
        labels = classify_candidates_with_llm(query_text, candidate_nodes, G)
        for n in candidate_nodes:
            X.append(G.nodes[n]['embedding'])
            y.append(labels[n])

    X = torch.tensor(np.stack(X), dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)
    torch.save((X, y), TRAIN_DATA_PATH)
    print(f"✅ GNN训练数据生成完成，保存至 {TRAIN_DATA_PATH}")

if __name__ == "__main__":
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    with open(QUERIES_PATH, "rb") as f:
        queries = pickle.load(f)

    generate_training_data(G, queries, topk=5)