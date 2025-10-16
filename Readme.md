# 智慧医疗 KGQA 系统

本项目实现了一个基于知识图谱 (KG) + GNN + RAG 的智慧医疗问答系统。系统通过 LLM 抽取医学知识构建 KG，再训练 GNN 对候选实体进行二分类，最后结合 RAG 进行问答推理。

---

## 目录结构

```
project_root/
├── build_kg.py            # 用 LLM 构建医学知识图谱并生成节点 embedding
├── gen_gnn_queries.py     # 根据 KG 文本生成训练用 query
├── gen_gnn_train_data.py  # 根据 query + PPR + LLM 生成 GNN 训练数据 (X, y)
├── train_gnn.py           # 使用训练数据训练 GNN 模型
├── rag_inference.py       # 使用 GNN + RAG 进行问答推理
├── data/medical_knowledge.txt  # 医学文本数据，每行一个片段
├── graph.pkl              # 保存的知识图谱
├── gnn_queries.pkl        # 保存的训练 query
├── gnn_train_data.pt      # GNN 训练数据
├── gnn_model.pt           # 训练完成的 GNN 模型
└── README.md
```

---

## 安装依赖

```
pip install torch torch_geometric networkx sentence-transformers tqdm openai
```

> 注意：`torch_geometric` 安装可能需要参考官方文档，根据 PyTorch 版本选择对应的安装命令。

---

## 配置参数

在各个文件中使用的配置参数：

```python
EMBED_MODEL = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6081/v1"
)
```

> 使用本地部署的 OpenAI 接口或者 ChatAnywhere 接口。embedding 模型路径为本地 sentence-transformers 模型。

---

## 使用流程

### 1. 构建知识图谱

```
python build_kg.py
```

* 从 `data/medical_knowledge.txt` 中读取医学文本
* 使用 LLM 抽取实体与关系构建知识图谱
* 为每个节点生成 embedding
* 输出 `graph.pkl`

### 2. 生成 GNN 训练用 query

```
python gen_gnn_queries.py
```

* 遍历知识图谱节点文本
* 使用 LLM 生成对应的问答 query
* 输出 `gnn_queries.pkl`

### 3. 生成 GNN 训练数据

```
python gen_gnn_train_data.py
```

* 对每个 query 使用 PPR 获取候选实体
* 使用 LLM 对候选实体进行二分类（是否答案）
* 输出 `(X, y)` 数据保存为 `gnn_train_data.pt`

### 4. 训练 GNN

```
python train_gnn.py
```

* 使用训练数据 `(X, y)` 训练 GCN 模型
* 输出训练完成的 GNN 模型 `gnn_model.pt`

### 5. RAG 推理问答

```
python rag_inference.py
```

* 输入 query，计算与 KG 节点相似度
* 使用 GNN 对候选节点进行评分
* 结合 LLM 生成最终答案

---

## 文件说明

* **build_kg.py**: 构建医学知识图谱并生成节点 embedding
* **gen_gnn_queries.py**: 生成训练用 query，输出 JSON 列表
* **gen_gnn_train_data.py**: 使用 PPR + LLM 生成 GNN 训练数据 `(X, y)`
* **train_gnn.py**: 训练 GNN 模型
* **rag_inference.py**: 使用 GNN + RAG 进行问答推理

---

## 注意事项

1. LLM 输出可能包含 ` ```json ` 或多行 JSON，需要在 `gen_gnn_queries.py` 中进行解析。
2. 训练 GNN 需要确保 `(X, y)` 数据与节点 embedding 对应。
3. 可以根据需要调整 PPR topk 参数和 GNN 超参数。
4. 建议先在小数据集上测试整个流程，确保接口和依赖正常。

---

## 联系

如有问题或改进建议，请联系项目维护者。
