import gradio as gr
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from train_gnn import GCN  # 导入 GCN 模型结构

# 配置和初始化
GRAPH_PATH = "graph.pkl"
MODEL_PATH = "gnn_model.pt"
EMBED_MODEL = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6081/v1"
)

# 全局变量，避免重复加载
_G = None
_gnn_model = None

def load_resources():
    """加载图和模型资源"""
    global _G, _gnn_model
    if _G is None or _gnn_model is None:
        print("正在加载图数据...")
        with open(GRAPH_PATH, "rb") as f:
            _G = pickle.load(f)
        
        # 获取节点 embedding 维度
        node_emb_dim = len(_G.nodes[list(_G.nodes())[0]]['embedding'])
        
        # 定义模型结构并加载参数
        print("正在加载GNN模型...")
        _gnn_model = GCN(in_dim=node_emb_dim, hidden_dim=64)
        _gnn_model.load_state_dict(torch.load(MODEL_PATH))
        print("资源加载完成！")
    
    return _G, _gnn_model

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

# 处理用户查询的函数
def process_query(query):
    if not query.strip():
        return "请输入有效的问题。"
    
    try:
        G, gnn_model = load_resources()
        answer = rag_query(query, G, gnn_model, topk=5)
        return answer
        
    except Exception as e:
        return f"处理查询时出现错误：{str(e)}"

# 创建Gradio界面
with gr.Blocks(
    title="智能医保问答系统 - 全国医保大赛",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="teal",
        neutral_hue="slate"
    ),
    css="""
    :root {
        --primary-color: #1e88e5;
        --secondary-color: #00acc1;
        --accent-color: #5e35b1;
        --background-color: #f8fafc;
        --card-background: #ffffff;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
    }
    
    .gradio-container {
        max-width: 1000px;
        margin: auto;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        min-height: 100vh;
        padding: 20px;
    }
    
    .main-header {
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(135deg, #1e88e5 0%, #0d47a1 100%);
        border-radius: 20px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(30, 136, 229, 0.3);
        color: white;
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        animation: float 20s linear infinite;
    }
    
    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-20px, -20px) rotate(360deg); }
    }
    
    .title {
        font-size: 2.8em;
        font-weight: 800;
        margin-bottom: 10px;
        background: linear-gradient(45deg, #ffffff, #e3f2fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.3em;
        font-weight: 300;
        opacity: 0.9;
        margin-bottom: 15px;
    }
    
    .tech-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        margin: 0 5px;
        backdrop-filter: blur(10px);
    }
    
    .input-section {
        background: var(--card-background);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        backdrop-filter: blur(10px);
    }
    
    .output-section {
        background: var(--card-background);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        backdrop-filter: blur(10px);
    }
    
    .input-box, .output-box {
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        font-size: 1em;
    }
    
    .input-box:focus, .input-box:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(30, 136, 229, 0.1);
    }
    
    .submit-btn {
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-size: 1.1em;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(30, 136, 229, 0.4);
    }
    
    .submit-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(30, 136, 229, 0.6);
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
    }
    
    .clear-btn {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-size: 1.1em;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }
    
    .clear-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(100, 116, 139, 0.4);
    }
    
    .status-ready {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
        border: none;
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
        border: none;
    }
    
    .status-processing {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.4);
        border: none;
    }
    
    .examples-section {
        background: var(--card-background);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    .example-chip {
        background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
        border: 1px solid #4fc3f7;
        border-radius: 25px;
        padding: 8px 16px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9em;
        color: var(--text-primary);
    }
    
    .example-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 195, 247, 0.3);
        background: linear-gradient(135deg, #b3e5fc 0%, #81d4fa 100%);
    }
    
    .footer {
        text-align: center;
        padding: 20px;
        color: var(--text-secondary);
        font-size: 0.9em;
        border-top: 1px solid #e2e8f0;
        margin-top: 20px;
    }
    
    .feature-icon {
        font-size: 2em;
        margin-bottom: 10px;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin: 25px 0;
    }
    
    .feature-card {
        background: var(--card-background);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.5);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    """
) as demo:
    
    # 主标题区域
    with gr.Column(elem_classes="main-header"):
        gr.HTML("""
        <div style="position: relative; z-index: 2;">
            <div style="font-size: 4em; margin-bottom: 20px;">🏥 💊</div>
            <div class="title">智能医保问答系统</div>
            <div class="subtitle">基于GNN+GraphRAG的下一代医学知识检索技术</div>
            <div style="margin-top: 20px;">
                <span class="tech-badge">图神经网络</span>
                <span class="tech-badge">GraphRAG</span>
                <span class="tech-badge">大语言模型</span>
                <span class="tech-badge">智能检索</span>
            </div>
        </div>
        """)
    
    # 特性展示
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🧠</div>
                    <h3 style="margin: 10px 0; color: var(--text-primary);">智能理解</h3>
                    <p style="color: var(--text-secondary); margin: 0;">深度理解医学问题语义</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <h3 style="margin: 10px 0; color: var(--text-primary);">精准检索</h3>
                    <p style="color: var(--text-secondary); margin: 0;">基于图结构的精准知识检索</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <h3 style="margin: 10px 0; color: var(--text-primary);">专业解答</h3>
                    <p style="color: var(--text-secondary); margin: 0;">生成专业可靠的医学答案</p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h3 style="margin: 10px 0; color: var(--text-primary);">高效响应</h3>
                    <p style="color: var(--text-secondary); margin: 0;">快速响应用户查询需求</p>
                </div>
            </div>
            """)
    
    # 输入区域
    with gr.Column(elem_classes="input-section"):
        gr.Markdown("## 💬 请输入您的医学问题")
        query_input = gr.Textbox(
            label="",
            placeholder="例如：医疗保险的报销流程是什么？糖尿病患者的医保政策有哪些？高血压药物是否在医保范围内？...",
            lines=4,
            max_lines=6,
            container=True,
            elem_classes="input-box",
            show_label=False
        )
        
        with gr.Row():
            submit_btn = gr.Button("🚀 智能分析", variant="primary", size="lg", elem_classes="submit-btn")
            clear_btn = gr.Button("🗑️ 清空内容", variant="secondary", size="lg", elem_classes="clear-btn")
    
    # 输出区域
    with gr.Column(elem_classes="output-section"):
        gr.Markdown("## 📋 专业解答")
        output = gr.Textbox(
            label="",
            lines=10,
            max_lines=20,
            show_copy_button=True,
            container=True,
            elem_classes="output-box",
            show_label=False
        )
    
    # 状态指示器 - 使用HTML组件来避免update问题
    status_display = gr.HTML(
        value="<div class='status-ready'>✅ 系统就绪 - 请输入您的医学问题</div>",
        label=""
    )
    
    # 示例问题
    with gr.Column(elem_classes="examples-section"):
        gr.Markdown("## 💡 快速提问")
        gr.Examples(
            examples=[
                ["医疗保险的报销比例是多少？"],
                ["糖尿病患者的医保报销政策？"],
                ["高血压常用药物是否在医保目录内？"],
                ["异地就医医保如何结算？"],
                ["门诊特殊疾病的医保待遇？"],
                ["医保个人账户的使用范围？"]
            ],
            inputs=query_input,
            label="点击以下问题快速体验",
            examples_per_page=6
        )
    
    # 底部信息
    gr.HTML("""
    <div class="footer">
        <p>💡 本系统基于GNN+GraphRAG技术构建，提供专业的医保知识问答服务</p>
        <p>⚠️ 提示：本系统提供的信息仅供参考，不能替代专业医疗建议。如有具体医疗问题，请咨询专业医生或医保部门。</p>
        <p style="margin-top: 15px; font-size: 0.8em; color: #94a3b8;">全国医保大赛参赛作品 | 智能医保问答系统 v1.0</p>
    </div>
    """)
    
    # 状态更新函数
    def get_status_html(status_text, status_type="ready"):
        if status_type == "ready":
            class_name = "status-ready"
        elif status_type == "processing":
            class_name = "status-processing"
        elif status_type == "error":
            class_name = "status-error"
        else:
            class_name = "status-ready"
        
        return f"<div class='{class_name}'>{status_text}</div>"
    
    # 绑定事件
    def submit_query(query):
        if not query.strip():
            return "", get_status_html("❌ 请输入有效的问题", "error")
        
        try:
            # 先返回处理中状态
            status_html = get_status_html("⏳ 正在智能分析您的问题...", "processing")
            
            # 执行查询
            G, gnn_model = load_resources()
            answer = rag_query(query, G, gnn_model, topk=5)
            
            # 查询完成
            return answer, get_status_html("✅ 分析完成 - 已生成专业解答", "ready")
            
        except Exception as e:
            error_msg = f"❌ 系统错误：{str(e)}"
            return f"处理查询时出现错误：{str(e)}", get_status_html(error_msg, "error")
    
    submit_btn.click(
        fn=submit_query,
        inputs=query_input,
        outputs=[output, status_display]
    )
    
    clear_btn.click(
        fn=lambda: ("", get_status_html("✅ 系统就绪 - 请输入您的医学问题", "ready")),
        inputs=[],
        outputs=[query_input, status_display]
    )
    
    # 回车键提交
    query_input.submit(
        fn=submit_query,
        inputs=query_input,
        outputs=[output, status_display]
    )

if __name__ == "__main__":
    # 预加载资源
    print("正在预加载资源...")
    load_resources()
    
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )