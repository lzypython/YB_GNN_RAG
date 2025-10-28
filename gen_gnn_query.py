import pickle
import openai
import os
from tqdm import tqdm

GRAPH_PATH = "graph_cx.pkl"
OUTPUT_QUERIES_PATH = "gnn_queries_cx.pkl"

client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6082/v1"
)

def generate_queries_from_text(texts, max_queries_per_text=3):
    queries = []
    for text in tqdm(texts, desc="生成训练问题"):
        prompt = f"""
        给定以下医学文本片段：
        \"\"\"{text}\"\"\"
        请生成最多 {max_queries_per_text} 个问句，每个问题对应的答案实体在文本中已经出现。
        输出格式：JSON 列表 [{{"question": "问题", "answer": "实体"}}]
        """

        try:
            resp = client.chat.completions.create(
                model="qwen2.5-7b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message.content
            # print("LLM 生成的问题:", content)
            content = content.replace("```", '')
            content = content.replace("json", '')
            import json
            data = json.loads(content)
            for q in data:
                queries.append(q)
        except Exception as e:
            print("生成问题失败，跳过:", e)
    return queries

if __name__ == "__main__":
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    texts = [G.nodes[n]["text"] for n in G.nodes()]
    queries = generate_queries_from_text(texts)

    with open(OUTPUT_QUERIES_PATH, "wb") as f:
        pickle.dump(queries, f)

    print(f"✅ 生成 {len(queries)} 个训练问题，保存至 {OUTPUT_QUERIES_PATH}")