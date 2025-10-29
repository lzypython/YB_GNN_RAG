import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import re
# 配置参数
EMBED_MODEL = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"
DATA_PATH = "knowledge_cx.txt"

# 初始化LLM客户端
client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6082/v1"
)


class SimpleRAG:
    def __init__(self):
        # 初始化嵌入模型
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.chunks = []
        self.chunk_embeddings = None
        
    def load_and_chunk_text(self, chunk_size=15, chunk_overlap=50):
        """加载文本并分块"""
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 简单的文本分块
        sentences = re.split(r'[。！？.!?]', text)
        self.chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    self.chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
                
        # 添加最后一个chunk
        if current_chunk:
            self.chunks.append(current_chunk.strip())
        
        print(f"总共分成了 {len(self.chunks)} 个chunks")
        
        # 生成嵌入向量
        self.chunk_embeddings = self.embed_model.encode(self.chunks)
        
    def retrieve_chunks(self, query, top_k=3):
        """根据query检索最相关的chunks"""
        query_embedding = self.embed_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.chunk_embeddings)[0]
        
        # 获取最相似的top_k个chunk
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        retrieved_chunks = [self.chunks[i] for i in top_indices]
        
        return retrieved_chunks, [similarities[i] for i in top_indices]
    
    def generate_response(self, query, top_k=3):
        """生成回答"""
        # 检索相关chunks
        retrieved_chunks, similarities = self.retrieve_chunks(query, top_k)
        
        # 构建prompt
        context = "\n\n".join(retrieved_chunks)
        prompt = f"""基于以下背景信息简单回答问题,不确定就说不知道，不要猜测。

        {context}

        问题：{query}
        回答："""
                
        # 调用LLM生成回答
        try:
            response = client.chat.completions.create(
                model="qwen2.5-7b",  # 根据你的本地模型调整
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # 返回结果和检索的chunks（用于调试）
            return {
                "answer": answer,
                "retrieved_chunks": retrieved_chunks,
                "similarities": similarities
            }
            
        except Exception as e:
            return {
                "answer": f"错误：{str(e)}",
                "retrieved_chunks": retrieved_chunks,
                "similarities": similarities
            }

# 初始化RAG系统
rag_system = SimpleRAG()

def init_rag_system():
    """初始化RAG系统"""
    print("正在加载文本和分块...")
    rag_system.load_and_chunk_text()
    print("RAG系统初始化完成！")

def rag_query(query, top_k=1):
    """
    输入query，返回LLM的回答
    
    参数:
    query: 用户问题
    top_k: 检索的chunk数量
    
    返回:
    dict: 包含回答和检索信息
    """
    return rag_system.generate_response(query, top_k)

# 使用示例
if __name__ == "__main__":

    # 初始化系统
    init_rag_system()
    qa_path = "medical_insurance_qa_cx.json"
    import json
    # 打开文件
    result_json = []
    import tqdm
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    for qa in tqdm.tqdm(qa_data):
        question = qa['question']
        answer = qa['answer']
        result = rag_query(question)
        qa["predicted_answer"] = result["answer"]
        result_json.append(qa)
    with open("rag_inference_results_cx_with_pred.json", 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)
    print("完成问答生成，结果已保存。")