import os
import pickle
import json
import re
import networkx as nx
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import openai

# ========= 配置 =========
DATA_PATH = "knowledge_cx.txt"
GRAPH_PATH = "graph_cx.pkl"
GRAPH_JSON_PATH = "graph_cx.json"
EMBED_MODEL = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

# 初始化 OpenAI 客户端
client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6082/v1"
)

# ========= 用 LLM 抽取实体和关系 =========
def extract_entities_relations_with_llm(text):
    prompt = f"""
从下面医学文本中抽取实体和关系：
文本：{text}

-Output Format-
对于每个实体：
("entity" <|> <entity_name> <|> <entity_type> <|> <source_text>)
对于每个关系：
("relationship" <|> <source_entity> <|> <target_entity> <|> <relationship_type> <|> <source_text>)

请严格按照上述格式输出，每行一个实体或关系。
"""
    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        result = response.choices[0].message.content
        entities = []
        relations = []
        for line in result.strip().split("\n"):
            line = line.strip()
            if line.startswith('("entity"'):
                parts = re.split(r"<\|>", line)
                if len(parts) == 4:
                    _, name, ent_type, src_text = [p.strip().strip('()"') for p in parts]
                    entities.append((name, ent_type, src_text))
            elif line.startswith('("relationship"'):
                parts = re.split(r"<\|>", line)
                if len(parts) == 5:
                    _, src, tgt, rel_type, src_text = [p.strip().strip('()"') for p in parts]
                    relations.append((src, tgt, rel_type, src_text))
        return entities, relations
    except Exception as e:
        print("LLM抽取失败，跳过该段文本:", e)
        return [], []

# 构建图
def build_graph(texts):
    G = nx.DiGraph()
    model = SentenceTransformer(EMBED_MODEL)
    for text in tqdm(texts, desc="构建知识图谱"):
        entities, relations = extract_entities_relations_with_llm(text)
        for name, ent_type, src_text in entities:
            if name not in G:
                G.add_node(name, text=src_text, type=ent_type, embedding=model.encode(name).tolist())
        for src, tgt, rel_type, src_text in relations:
            if src not in G:
                G.add_node(src, text=src_text, type='Unknown', embedding=model.encode(src).tolist())
            if tgt not in G:
                G.add_node(tgt, text=src_text, type='Unknown', embedding=model.encode(tgt).tolist())
            G.add_edge(src, tgt, relation=rel_type, source_text=src_text)
    return G

# 将图转换为 JSON 保存
def save_graph_json(G, json_path):
    data = {
        "nodes": [],
        "edges": []
    }
    for n, attr in G.nodes(data=True):
        data["nodes"].append({
            "id": n,
            "text": attr.get("text", ""),
            "type": attr.get("type", ""),
            # "embedding": attr.get("embedding", [])
        })
    for u, v, attr in G.edges(data=True):
        data["edges"].append({
            "source": u,
            "target": v,
            "relation": attr.get("relation", ""),
            "source_text": attr.get("source_text", "")
        })
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    G = build_graph(docs)
    # 保存 pickle
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    # 保存 JSON
    save_graph_json(G, GRAPH_JSON_PATH)

    print(f"✅ 知识图谱构建完成，节点数={len(G.nodes())}，边数={len(G.edges())}")
    print(f"✅ JSON 文件保存至 {GRAPH_JSON_PATH}")
