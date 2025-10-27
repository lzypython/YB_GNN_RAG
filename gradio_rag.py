import gradio as gr
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from train_gnn import GCN  # 导入 GCN 模型结构
import datetime
import random

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

# 对话历史存储
conversation_history = []

# 统计数据
stats = {
    "total_queries": 0,
    "today_queries": 0
}

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

def create_message_html(message, is_user=False):
    """创建消息HTML"""
    timestamp = datetime.datetime.now().strftime("%H:%M")
    if is_user:
        return f"""
        <div class="message user-message">
            <div class="message-content">
                <div class="message-text">{message}</div>
                <div class="message-time">{timestamp}</div>
            </div>
            <div class="message-avatar">
                <div class="avatar-icon">👤</div>
            </div>
        </div>
        """
    else:
        return f"""
        <div class="message bot-message">
            <div class="message-avatar">
                <div class="avatar-icon">🏥</div>
            </div>
            <div class="message-content">
                <div class="message-text">{message}</div>
                <div class="message-time">{timestamp}</div>
            </div>
        </div>
        """

def submit_query(query, chat_history):
    """处理用户查询并更新对话历史"""
    global stats
    if not query.strip():
        return "", chat_history, get_status_html("❌ 请输入有效的问题", "error"), update_stats_display()
    
    try:
        # 更新统计
        stats["total_queries"] += 1
        stats["today_queries"] += 1
        
        # 添加用户消息到历史
        user_message_html = create_message_html(query, is_user=True)
        if "chat-placeholder" in chat_history:
            chat_history = f'<div class="chat-messages-wrapper">{user_message_html}'
        else:
            chat_history = chat_history.replace('</div>', '') + user_message_html
        
        # 更新状态
        status_html = get_status_html("⏳ AI正在分析您的问题...", "processing")
        
        # 执行查询
        G, gnn_model = load_resources()
        answer = rag_query(query, G, gnn_model, topk=5)
        
        # 添加AI回复到历史
        bot_message_html = create_message_html(answer, is_user=False)
        chat_history += bot_message_html + '</div>'
        
        # 更新状态
        status_html = get_status_html("✅ 已回答您的问题", "ready")
        
        return "", chat_history, status_html, update_stats_display()
        
    except Exception as e:
        error_msg = f"❌ 系统错误：{str(e)}"
        bot_message_html = create_message_html(f"抱歉，处理您的查询时出现了错误：{str(e)}", is_user=False)
        if "chat-placeholder" in chat_history:
            chat_history = f'<div class="chat-messages-wrapper">{bot_message_html}</div>'
        else:
            chat_history = chat_history.replace('</div>', '') + bot_message_html + '</div>'
        return "", chat_history, get_status_html(error_msg, "error"), update_stats_display()

def get_status_html(status_text, status_type="ready"):
    """生成状态HTML"""
    icons = {
        "ready": "✅",
        "processing": "⏳", 
        "error": "❌"
    }
    icon = icons.get(status_type, "✅")
    
    return f"""
    <div class="status-indicator status-{status_type}">
        <div class="status-icon">{icon}</div>
        <div class="status-text">{status_text}</div>
    </div>
    """

def update_stats_display():
    """更新统计信息显示"""
    return f"""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-icon">📊</div>
            <div class="stat-content">
                <div class="stat-value">{stats['total_queries']}</div>
                <div class="stat-label">总查询量</div>
            </div>
        </div>
        <div class="stat-item">
            <div class="stat-icon">📅</div>
            <div class="stat-content">
                <div class="stat-value">{stats['today_queries']}</div>
                <div class="stat-label">今日查询</div>
            </div>
        </div>
    </div>
    """

def clear_chat():
    """清空对话历史"""
    return get_chat_placeholder(), get_status_html("✅ 对话已清空，开始新的对话", "ready"), update_stats_display()

def get_chat_placeholder():
    """获取聊天区域占位符"""
    return """
    <div class="chat-placeholder">
        <div class="placeholder-icon">💬</div>
        <div class="placeholder-title">欢迎使用智能医保问答系统</div>
        <div class="placeholder-subtitle">基于GNN+GraphRAG的下一代医学知识检索技术</div>
        <div class="placeholder-features">
            <div class="feature-item">🔍 精准医学知识检索</div>
            <div class="feature-item">💊 专业医保政策解读</div>
            <div class="feature-item">🏥 智能疾病分析</div>
            <div class="feature-item">📈 实时数据更新</div>
        </div>
    </div>
    """

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
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --gradient-accent: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    .gradio-container {
        max-width: 1400px;
        margin: auto;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 50%, #f1f5f9 100%);
        min-height: 100vh;
        padding: 0;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    .main-header {
        background: var(--gradient-primary);
        padding: 40px 0;
        margin-bottom: 0;
        border-radius: 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" opacity="0.1"><polygon points="0,0 1000,50 1000,100 0,100" fill="white"/></svg>');
        background-size: cover;
    }
    
    .header-content {
        position: relative;
        z-index: 2;
        text-align: center;
        color: white;
    }
    
    .title {
        font-size: 3.2em;
        font-weight: 800;
        margin-bottom: 15px;
        background: linear-gradient(45deg, #ffffff, #e3f2fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 4px 20px rgba(0,0,0,0.2);
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        font-size: 1.4em;
        font-weight: 300;
        opacity: 0.95;
        margin-bottom: 25px;
        letter-spacing: 0.5px;
    }
    
    .tech-badges {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        margin-top: 25px;
    }
    
    .tech-badge {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(20px);
        padding: 10px 20px;
        border-radius: 25px;
        font-size: 0.9em;
        font-weight: 500;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .tech-badge:hover {
        background: rgba(255,255,255,0.25);
        transform: translateY(-2px);
    }
    
    .main-content {
        display: grid;
        grid-template-columns: 300px 1fr 300px;
        gap: 25px;
        padding: 25px;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .sidebar {
        display: flex;
        flex-direction: column;
        gap: 25px;
    }
    
    .stats-card {
        background: var(--card-background);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        backdrop-filter: blur(10px);
    }
    
    .stats-container {
        display: flex;
        flex-direction: column;
        gap: 15px;
    }
    
    .stat-item {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 15px;
        background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        border-radius: 15px;
        border: 1px solid #e2e8f0;
    }
    
    .stat-icon {
        font-size: 1.5em;
        width: 50px;
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--gradient-accent);
        border-radius: 12px;
    }
    
    .stat-content {
        flex: 1;
    }
    
    .stat-value {
        font-size: 1.5em;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    
    .stat-label {
        font-size: 0.85em;
        color: var(--text-secondary);
        margin-top: 4px;
    }
    
    .features-card {
        background: var(--card-background);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        backdrop-filter: blur(10px);
    }
    
    .features-title {
        font-size: 1.2em;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .feature-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        background: #f8fafc;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .feature-item:hover {
        background: #e2e8f0;
        transform: translateX(5px);
    }
    
    .system-info-card {
        background: var(--card-background);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        backdrop-filter: blur(10px);
    }
    
    .system-info-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
    }
    
    .system-info-item:last-child {
        border-bottom: none;
    }
    
    .system-info-icon {
        font-size: 1.2em;
        width: 30px;
        text-align: center;
    }
    
    .system-info-content {
        flex: 1;
    }
    
    .system-info-label {
        font-size: 0.85em;
        color: var(--text-secondary);
        margin-bottom: 2px;
    }
    
    .system-info-value {
        font-size: 0.95em;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .chat-area {
        display: flex;
        flex-direction: column;
        gap: 25px;
    }
    
    .chat-container {
        background: var(--card-background);
        border-radius: 25px;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        border: 1px solid rgba(255,255,255,0.6);
        backdrop-filter: blur(15px);
        height: 650px;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        position: relative;
    }
    
    .chat-messages-container {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        position: relative;
    }
    
    .chat-messages-wrapper {
        flex: 1;
        overflow-y: auto;
        padding: 25px;
        display: flex;
        flex-direction: column;
        gap: 20px;
        background: linear-gradient(180deg, #fafbfc 0%, #ffffff 100%);
        max-height: 100%;
        scroll-behavior: smooth;
    }
    
    .message {
        display: flex;
        align-items: flex-start;
        gap: 15px;
        animation: messageSlide 0.4s ease-out;
    }
    
    @keyframes messageSlide {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .user-message {
        justify-content: flex-end;
    }
    
    .bot-message {
        justify-content: flex-start;
    }
    
    .message-avatar {
        width: 45px;
        height: 45px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .user-message .message-avatar {
        background: var(--gradient-secondary);
        order: 2;
    }
    
    .bot-message .message-avatar {
        background: var(--gradient-accent);
    }
    
    .avatar-icon {
        font-size: 1.3em;
    }
    
    .message-content {
        max-width: 65%;
        padding: 18px 22px;
        border-radius: 20px;
        position: relative;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    .user-message .message-content {
        background: var(--gradient-secondary);
        color: white;
        border-bottom-right-radius: 6px;
    }
    
    .bot-message .message-content {
        background: white;
        border: 1px solid #f1f5f9;
        border-bottom-left-radius: 6px;
    }
    
    .message-text {
        line-height: 1.6;
        font-size: 0.95em;
        white-space: pre-wrap;
    }
    
    .message-time {
        font-size: 0.75em;
        opacity: 0.7;
        margin-top: 8px;
        text-align: right;
    }
    
    .input-section {
        background: var(--card-background);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        backdrop-filter: blur(10px);
    }
    
    .input-box {
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        font-size: 1em;
        padding: 18px 20px;
        background: #fafbfc;
    }
    
    .input-box:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 4px rgba(30, 136, 229, 0.15);
        background: white;
    }
    
    .submit-btn {
        background: var(--gradient-primary);
        border: none;
        border-radius: 15px;
        padding: 16px 32px;
        font-size: 1.1em;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .submit-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.6);
    }
    
    .clear-btn {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%);
        border: none;
        border-radius: 15px;
        padding: 16px 32px;
        font-size: 1.1em;
        font-weight: 600;
        color: white;
        transition: all 0.3s ease;
    }
    
    .clear-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(100, 116, 139, 0.4);
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 15px 20px;
        border-radius: 15px;
        font-weight: 600;
        margin-bottom: 20px;
    }
    
    .status-ready {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    
    .status-processing {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    
    .status-icon {
        font-size: 1.2em;
    }
    
    .examples-section {
        background: var(--card-background);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.5);
        backdrop-filter: blur(10px);
    }
    
    .examples-title {
        font-size: 1.2em;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .examples-grid {
        display: grid;
        grid-template-columns: 1fr;
        gap: 12px;
    }
    
    .example-chip {
        background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
        border: 1px solid #4fc3f7;
        border-radius: 15px;
        padding: 12px 16px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9em;
        color: var(--text-primary);
        text-align: left;
    }
    
    .example-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 195, 247, 0.3);
        background: linear-gradient(135deg, #b3e5fc 0%, #81d4fa 100%);
    }
    
    .chat-placeholder {
        text-align: center;
        color: #64748b;
        padding: 80px 40px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background: linear-gradient(180deg, #fafbfc 0%, #ffffff 100%);
    }
    
    .placeholder-icon {
        font-size: 4em;
        margin-bottom: 25px;
        opacity: 0.6;
    }
    
    .placeholder-title {
        font-size: 1.8em;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 15px;
    }
    
    .placeholder-subtitle {
        font-size: 1.1em;
        color: var(--text-secondary);
        margin-bottom: 30px;
        max-width: 400px;
        line-height: 1.5;
    }
    
    .placeholder-features {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 15px;
        max-width: 400px;
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 15px;
        background: #f8fafc;
        border-radius: 10px;
        font-size: 0.9em;
    }
    
    .footer {
        text-align: center;
        padding: 30px 25px;
        color: var(--text-secondary);
        font-size: 0.9em;
        border-top: 1px solid #e2e8f0;
        margin-top: 25px;
        background: white;
    }
    
    .footer-content {
        max-width: 800px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    .button-group {
        display: flex;
        gap: 12px;
        margin-top: 20px;
    }
    
    /* 滚动条样式优化 */
    .chat-messages-wrapper::-webkit-scrollbar {
        width: 8px;
    }
    
    .chat-messages-wrapper::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    .chat-messages-wrapper::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
        transition: background 0.3s ease;
    }
    
    .chat-messages-wrapper::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    .chat-messages-wrapper {
        scrollbar-width: thin;
        scrollbar-color: #cbd5e1 #f1f5f9;
    }
    
    /* 自动滚动到底部的样式 */
    .auto-scroll {
        scroll-behavior: smooth;
    }
    
    /* 消息容器样式优化 */
    .message:last-child {
        margin-bottom: 0;
    }
    
    /* 响应式设计 */
    @media (max-width: 1200px) {
        .main-content {
            grid-template-columns: 280px 1fr 280px;
            gap: 20px;
            padding: 20px;
        }
        
        .chat-container {
            height: 600px;
        }
    }
    
    @media (max-width: 1024px) {
        .main-content {
            grid-template-columns: 1fr;
            gap: 20px;
        }
        
        .sidebar {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        .chat-container {
            height: 500px;
        }
    }
    
    @media (max-width: 768px) {
        .main-content {
            padding: 15px;
        }
        
        .sidebar {
            grid-template-columns: 1fr;
        }
        
        .chat-container {
            height: 450px;
        }
        
        .message-content {
            max-width: 80%;
        }
    }
    """
) as demo:
    
    # 主标题区域
    with gr.Column(elem_classes="main-header"):
        gr.HTML("""
        <div class="header-content">
            <div style="font-size: 4.5em; margin-bottom: 20px; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.2));">🏥 💊</div>
            <div class="title">智能医保问答系统</div>
            <div class="subtitle">全国医保大赛 - 基于GNN+GraphRAG的智能医学知识平台</div>
            <div class="tech-badges">
                <span class="tech-badge">图神经网络</span>
                <span class="tech-badge">GraphRAG技术</span>
                <span class="tech-badge">大语言模型</span>
                <span class="tech-badge">智能语义检索</span>
                <span class="tech-badge">实时知识图谱</span>
            </div>
        </div>
        """)
    
    # 主要内容区域 - 三栏布局
    with gr.Row(elem_classes="main-content"):
        
        # 左侧边栏 - 统计和功能
        with gr.Column(elem_classes="sidebar"):
            
            # 统计信息卡片
            stats_display = gr.HTML(
                value=update_stats_display(),
                label=""
            )
            
            # 系统特性卡片
            gr.HTML("""
            <div class="features-card">
                <div class="features-title">
                    <span>🚀</span> 系统特性
                </div>
                <div class="feature-list">
                    <div class="feature-item">
                        <span>🔍</span>
                        <span>精准医学检索</span>
                    </div>
                    <div class="feature-item">
                        <span>💊</span>
                        <span>医保政策解读</span>
                    </div>
                    <div class="feature-item">
                        <span>🏥</span>
                        <span>疾病智能分析</span>
                    </div>
                    <div class="feature-item">
                        <span>📈</span>
                        <span>实时数据更新</span>
                    </div>
                    <div class="feature-item">
                        <span>🛡️</span>
                        <span>专业可靠回答</span>
                    </div>
                </div>
            </div>
            """)
            
            # 新增：系统信息卡片
            gr.HTML("""
            <div class="system-info-card">
                <div class="features-title">
                    <span>⚙️</span> 系统信息
                </div>
                <div class="feature-list">
                    <div class="system-info-item">
                        <div class="system-info-icon">🕐</div>
                        <div class="system-info-content">
                            <div class="system-info-label">启动时间</div>
                            <div class="system-info-value">""" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + """</div>
                        </div>
                    </div>
                    <div class="system-info-item">
                        <div class="system-info-icon">💻</div>
                        <div class="system-info-content">
                            <div class="system-info-label">运行状态</div>
                            <div class="system-info-value">正常运行</div>
                        </div>
                    </div>
                    <div class="system-info-item">
                        <div class="system-info-icon">📚</div>
                        <div class="system-info-content">
                            <div class="system-info-label">知识库版本</div>
                            <div class="system-info-value">v2.1.0</div>
                        </div>
                    </div>
                    <div class="system-info-item">
                        <div class="system-info-icon">🛠️</div>
                        <div class="system-info-content">
                            <div class="system-info-label">最后更新</div>
                            <div class="system-info-value">2024-01-15</div>
                        </div>
                    </div>
                </div>
            </div>
            """)
        
        # 中间区域 - 对话界面
        with gr.Column(elem_classes="chat-area"):
            
            # 状态指示器
            status_display = gr.HTML(
                value=get_status_html("✅ 系统就绪，请输入您的医学问题", "ready"),
                label=""
            )
            
            # 对话容器 - 固定高度滚动区域
            with gr.Column(elem_classes="chat-container"):
                chat_display = gr.HTML(
                    value=get_chat_placeholder(),
                    label="",
                    elem_id="chat-display"
                )
            
            # 输入区域
            with gr.Column(elem_classes="input-section"):
                with gr.Row():
                    query_input = gr.Textbox(
                        label="",
                        placeholder="💬 请输入您的医学问题，例如：医疗保险的报销流程、疾病预防措施、医保政策咨询等...",
                        lines=2,
                        max_lines=4,
                        container=True,
                        elem_classes="input-box",
                        show_label=False,
                        scale=4
                    )
                    with gr.Column(scale=1):
                        with gr.Row():
                            submit_btn = gr.Button("🚀 发送", variant="primary", size="lg", elem_classes="submit-btn")
                            clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="lg", elem_classes="clear-btn")
        
        # 右侧边栏 - 补充信息
        with gr.Column(elem_classes="sidebar"):
            
            # 技术架构卡片
            gr.HTML("""
            <div class="features-card">
                <div class="features-title">
                    <span>🏗️</span> 技术架构
                </div>
                <div class="feature-list">
                    <div class="feature-item">
                        <span>🧠</span>
                        <span>GNN图神经网络</span>
                    </div>
                    <div class="feature-item">
                        <span>🔗</span>
                        <span>GraphRAG检索</span>
                    </div>
                    <div class="feature-item">
                        <span>🤖</span>
                        <span>智能语言模型</span>
                    </div>
                    <div class="feature-item">
                        <span>📚</span>
                        <span>医学知识图谱</span>
                    </div>
                    <div class="feature-item">
                        <span>⚡</span>
                        <span>实时推理引擎</span>
                    </div>
                </div>
            </div>
            """)
            
            # 使用说明卡片
            gr.HTML("""
            <div class="features-card">
                <div class="features-title">
                    <span>📋</span> 使用指南
                </div>
                <div class="feature-list">
                    <div class="feature-item">
                        <span>1.</span>
                        <span>输入医学相关问题</span>
                    </div>
                    <div class="feature-item">
                        <span>2.</span>
                        <span>系统智能分析检索</span>
                    </div>
                    <div class="feature-item">
                        <span>3.</span>
                        <span>获取专业解答</span>
                    </div>
                    <div class="feature-item">
                        <span>💡</span>
                        <span>支持多轮对话</span>
                    </div>
                </div>
            </div>
            """)
            
            # 示例问题卡片
            with gr.Column(elem_classes="examples-section"):
                gr.Markdown("### 💡 快速提问")
                examples = gr.Examples(
                    examples=[
                        ["医疗保险报销比例是多少？"],
                        ["糖尿病患者的医保政策？"], 
                        ["高血压药物医保目录？"],
                        ["异地就医如何结算？"],
                        ["门诊特殊疾病待遇？"],
                        ["医保个人账户使用范围？"]
                    ],
                    inputs=query_input,
                    label="点击以下问题快速体验",
                    examples_per_page=6
                )
    
    # 底部信息
    gr.HTML("""
    <div class="footer">
        <div class="footer-content">
            <p>💡 本系统基于先进的GNN+GraphRAG技术构建，提供专业、准确的医保知识问答服务</p>
            <p>⚠️ 重要提示：本系统提供的信息仅供参考，不能替代专业医疗建议。如有具体医疗问题，请咨询专业医生或医保部门。</p>
            <p style="margin-top: 20px; font-size: 0.85em; color: #94a3b8;">
                全国医保大赛参赛作品 | 智能医保问答系统 v2.0 | 技术支持：GNN + GraphRAG + 智能语言模型
            </p>
        </div>
    </div>
    """)
    
    # 绑定事件
    submit_btn.click(
        fn=submit_query,
        inputs=[query_input, chat_display],
        outputs=[query_input, chat_display, status_display, stats_display]
    )
    
    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chat_display, status_display, stats_display]
    )
    
    # 回车键提交
    query_input.submit(
        fn=submit_query,
        inputs=[query_input, chat_display],
        outputs=[query_input, chat_display, status_display, stats_display]
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