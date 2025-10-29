import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from train_gnn import GCN  # 导入 GCN 模型结构

GRAPH_PATH = "graph_cx.pkl"
MODEL_PATH = "gnn_model_cx.pt"
EMBED_MODEL = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6082/v1"
)
from rag_inference import rag_query,rag_query_cx

if __name__ == "__main__":
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    # 获取节点 embedding 维度
    node_emb_dim = len(G.nodes[list(G.nodes())[0]]['embedding'])

    # 定义模型结构并加载参数
    gnn_model = GCN(in_dim=node_emb_dim, hidden_dim=64)
    gnn_model.load_state_dict(torch.load(MODEL_PATH))
    import json
    qa_json = "medical_insurance_qa_cx.json"
    with open(qa_json, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    result = []
    import tqdm
    for qa in tqdm.tqdm(qa_data):
        query = qa.get("question", "")
        ans = rag_query_cx(query, G, gnn_model, topk=5)
        qa["predicted_answer"] = ans
        result.append(qa)
    output_file = "rag_inference_results_cx.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ RAG 推理完成，结果保存至 {output_file}")