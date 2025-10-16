# 智慧医疗 KGQA 系统

本项目实现了一个基于知识图谱 (KG) + GNN + GraphRAG 的智慧医疗问答系统。系统通过 LLM 抽取医学知识构建 KG，再训练 GNN 对候选实体进行s分类，最后结合 GraphRAG 进行问答推理。

---

## 目录结构

```
project_root/
├── build_kg.py            # 用 LLM 构建医学知识图谱并生成节点 embedding
├── gen_gnn_queries.py     # 根据 KG 文本生成训练用 query
├── gen_gnn_train_data.py  # 根据 query + PPR + LLM 生成 GNN 训练数据 (X, y)
├── train_gnn.py           # 使用训练数据训练 GNN 模型
├── rag_inference.py       # 使用 GNN + GraphRAG 进行问答推理
├── knowledge.txt          # 医学文本数据
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


## 测试样例

```
问：医疗保险是什么？
```
```
答：医疗保险是一种金融产品，旨在为被保险人提供医疗费用补偿或医疗服务，以减轻因疾病或意外伤害导致的经济负担。它通过合同形式规定了保险公司和被保险人之间的权利与义务。医疗保险是个人健康管理的重要组成部分，也是构建全面健康保障体系的关键环节。

### 医疗保险的主要功能包括：

1. **费用补偿**：当被保险人因疾病或意外伤害需要接受治疗时，保险公司会根据合同约定支付相应的医疗费用，如门诊费、住院费、手术费、药品费等。
2. **服务提供**：部分医疗保险产品还可能包含医疗服务，如预约专家门诊、安排住院床位、提供第二诊疗意见等。
3. **预防保健**：一些高端医疗保险还包括定期体检、健康咨询等预防性服务，帮助被保险人提前发现潜在健康问题并采取措施。

### 选择医疗保险时需要注意的几个方面：

- **保障范围**：了解保险产品覆盖的具体医疗服务项目，是否包括非社保目录内的治疗项目。这直接影响到保险的实际价值。
- **免责条款**：明确哪些情况不在保险赔偿范围内，比如既往病史、某些特定疾病的治疗等。
- **续保条件**：了解保险到期后是否容易续保，以及续保时是否有额外要求或加价。
- **理赔流程**：熟悉理赔申请所需材料及流程，确保在需要时能够顺利获得赔偿。

### 购买医疗保险的实用建议：

1. **需求分析**：根据自身健康状况、家庭成员情况及经济能力确定合适的保障额度和类型。
2. **比较选择**：多比较不同保险公司提供的产品，关注其保障范围、保费、免赔额等因素。
3. **仔细阅读条款**：特别是保障责任、免责条款、续保条件等内容，确保理解清楚后再做决定。
4. **咨询专业人士**：如有疑问，可咨询保险代理人或相关专业人士获取更多信息。

总之，医疗保险是个人和家庭健康管理的重要工具之一，合理配置医疗保险可以有效降低因疾病带来的经济风险，为健康生活提供坚实的保障。
```
