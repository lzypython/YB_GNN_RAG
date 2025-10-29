from bert_score import score
import json
import csv
def get_bert_score(Output_json: str,lang="en"):
    """
    读取结果 JSON 并计算平均 BERTScore - 优化版本
    """
    with open(Output_json, "r", encoding="utf-8") as f:
        results = json.load(f)
    references = [item["answer"] for item in results]
    predictions = [item["predicted_answer"] for item in results]
    
    scores = bert_score_batch(references, predictions, lang=lang)
    result = round(sum(scores) / len(scores), 4) * 100
    # print(f"MLLM回答平均BERTScore: {result}")
    return result

def bert_score_batch(references: list, predictions: list, lang="en") -> list:
    """
    批量计算BERTScore，显著提高速度
    """
    # 过滤掉空答案
    valid_indices = []
    valid_refs = []
    valid_preds = []
    
    for i, (ref, pred) in enumerate(zip(references, predictions)):
        if pred != "I don't know.":
            valid_indices.append(i)
            valid_refs.append(ref)
            valid_preds.append(pred)
    
    if not valid_refs:
        return [0.0] * len(references)
    
    # 批量计算
    P, R, F1 = score(valid_preds, valid_refs, lang=lang, batch_size=32, nthreads=4)
    f1_scores = F1.tolist()
    
    # 重建完整的结果列表
    results = []
    valid_idx = 0
    for i in range(len(references)):
        if i in valid_indices:
            results.append(float(f1_scores[valid_idx]))
            valid_idx += 1
        else:
            results.append(0.0)
    
    return results
import json
import openai

# 简化版本
def get_llm_score(json_file_path):
    """简化版的评估函数"""
    client = openai.OpenAI(
        api_key="empty",
        base_url="http://localhost:6082/v1"
    )
    
    # 读取数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        data = [data]
    
    scores = []
    
    for item in data:
        reference = item.get('answer', '')
        prediction = item.get('predicted_answer', '')
        
        if reference and prediction:
            prompt = f"""
            您是答案相关性评估员。给定一个参考答案和一个模型答案，输出一个介于0和1之间的浮动分数。回答的越全面，信息含量越丰富，分数越高；格式越规范，分数越高。

            Reference Answer: {reference}
            Model Answer: {prediction}

            Output only the score:
            """
            try:
                response = client.chat.completions.create(
                    model="qwen2.5-7b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                
                score_text = response.choices[0].message.content.strip()
                # 简单提取数字
                try:
                    score = float(score_text)
                    if 0 <= score <= 1:
                        scores.append(score)
                except:
                    pass
                    
            except Exception as e:
                # print(f"评估出错: {e}")
                pass
    
    if scores:
        avg_score = sum(scores) / len(scores)
        # print(f"评估了 {len(scores)} 个问答对")
        # print(f"平均评分: {avg_score:.4f}")
        return avg_score
    else:
        print("无法计算评分")
        return 0

# 使用
# average_score = simple_evaluate("your_file.json")
jsonpath_yw_YB = "/back-up/lzy/YB_GNN_RAG/rag_inference_results_yw.json"
jsonpath_yw_graphrag = "/back-up/lzy/graphrag/graphrag_yw/graphrag_results_yw.json"
jsonpath_yw_rag = "/back-up/lzy/YB_GNN_RAG/rag_inference_results_yw_with_pred.json"
yw_YB_bert = 0
yw_graphrag_bert = 0
yw_YB_llm = get_llm_score(jsonpath_yw_YB)
yw_graphrag_llm = get_llm_score(jsonpath_yw_graphrag)
yw_rag_llm = get_llm_score(jsonpath_yw_rag)
# print(f"YB_GNN_RAG 医疗保险数据集 LLM Score: {yw_YB_llm}")
# print(f"GraphRAG 医疗保险数据集 LLM Score: {yw_graphrag_llm}")
# print(f"YB_GNN_RAG 医疗保险数据集 BERTScore: {yw_YB_bert}")
# print(f"GraphRAG 医疗保险数据集 BERTScore: {yw_graphrag_bert}")
# 保存结果到 CSV
with open("evaluation_comparison_yw2.csv", "w", newline='', encoding="utf-8") as csvfile:
    fieldnames = ["Model", "LLM_Score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"Model": "RAG", "LLM_Score": yw_rag_llm})
    writer.writerow({"Model": "GraphRAG", "LLM_Score": yw_graphrag_llm})
    writer.writerow({"Model": "YB_GNN_RAG", "LLM_Score": yw_YB_llm})

# cx得分
jsonpath_cx_YB = "/back-up/lzy/YB_GNN_RAG/rag_inference_results_cx.json"
jsonpath_cx_graphrag = "/back-up/lzy/graphrag/graphrag_cx/graphrag_results_cx.json"
jsonpath_cx_rag = "/back-up/lzy/YB_GNN_RAG/rag_inference_results_cx_with_pred.json"
cx_YB_bert = 0
cx_graphrag_bert = 0
cx_YB_llm = get_llm_score(jsonpath_cx_YB)
cx_graphrag_llm = get_llm_score(jsonpath_cx_graphrag)
cx_rag_llm = get_llm_score(jsonpath_cx_rag)
# print(f"YB_GNN_RAG 疾病查询数据集 LLM Score: {cx_YB_llm}")
# print(f"GraphRAG 疾病查询数据集 LLM Score: {cx_graphrag_llm}")
# print(f"Y B_GNN_RAG 疾病查询数据集 BERTScore: {cx_YB_bert}")
# print(f"GraphRAG 疾病查询数据集 BERTScore: {cx_graphrag_bert}")
# 保存结果到 CSV
with open("evaluation_comparison_cx2.csv", "w", newline='', encoding="utf-8") as csvfile:
    fieldnames = ["Model", "LLM_Score"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({"Model": "RAG", "LLM_Score": cx_rag_llm})
    writer.writerow({"Model": "GraphRAG", "LLM_Score": cx_graphrag_llm})
    writer.writerow({"Model": "YB_GNN_RAG", "LLM_Score": cx_YB_llm})
print("评估完成，结果已保存到 CSV 文件。")