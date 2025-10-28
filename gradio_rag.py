import gradio as gr
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from train_gnn import GCN  # 导入 GCN 模型结构

# 配置和初始化
GRAPH_PATH = "graph_cx.pkl"
MODEL_PATH = "gnn_model_cx.pt"
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

from rag_inference import rag_query,rag_query_cx

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
    title="医学知识问答系统",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray"
    ),
    css="""
    .gradio-container {
        max-width: 800px;
        margin: auto;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
        background: linear-gradient(45deg, #1e88e5, #0d47a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .description {
        text-align: center;
        font-size: 1.1em;
        margin-bottom: 30px;
        color: #666;
    }
    .input-box {
        border-radius: 10px;
        padding: 20px;
    }
    .output-box {
        border-radius: 10px;
        padding: 20px;
        min-height: 200px;
    }
    .submit-btn {
        background: linear-gradient(45deg, #1e88e5, #0d47a1);
        border: none;
        color: white;
    }
    .submit-btn:hover {
        background: linear-gradient(45deg, #1565c0, #0d47a1);
    }
    """
) as demo:
    
    gr.Markdown(
        """
        # 🏥 医学知识问答系统
        
        <div class="description">
        基于图神经网络和语言模型的智能医学问答系统，为您提供专业的医学知识解答
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div style="text-align: center;">
                <div style="font-size: 80px; margin-bottom: 20px;">🏥</div>
            </div>
            """)
    
    with gr.Row():
        query_input = gr.Textbox(
            label="💬 请输入您的医学问题",
            placeholder="例如：医疗保险是什么？糖尿病有哪些症状？高血压如何预防？...",
            lines=3,
            max_lines=5,
            container=True,
            elem_classes="input-box"
        )
    
    with gr.Row():
        submit_btn = gr.Button("🚀 提交查询", variant="primary", size="lg", elem_classes="submit-btn")
        clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="lg")
    
    with gr.Row():
        output = gr.Textbox(
            label="📋 专业回答",
            lines=8,
            max_lines=15,
            show_copy_button=True,
            container=True,
            elem_classes="output-box"
        )
    
    # 状态指示器
    status = gr.Textbox(
        label="状态",
        value="✅ 系统就绪",
        interactive=False,
        max_lines=1
    )
    
    # 示例问题
    gr.Examples(
        examples=[
            ["医疗保险是什么？"],
            ["糖尿病的常见症状有哪些？"],
            ["高血压应该如何预防？"],
            ["心脏病的危险因素有哪些？"],
            ["感冒和流感的区别是什么？"]
        ],
        inputs=query_input,
        label="💡 示例问题（点击即可使用）"
    )
    
    # 底部信息
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #888; font-size: 0.9em;">
        💡 提示：本系统提供的信息仅供参考，不能替代专业医疗建议。如有医疗问题，请咨询专业医生。
        </div>
        """
    )
    
    # 绑定事件
    def submit_query(query):
        if not query.strip():
            return "请输入有效的问题。", "❌ 请输入问题"
        try:
            status_msg = "⏳ 正在处理您的查询..."
            G, gnn_model = load_resources()
            answer = rag_query(query, G, gnn_model, topk=5)
            return answer, "✅ 查询完成"
        except Exception as e:
            return f"处理查询时出现错误：{str(e)}", "❌ 处理错误"
    
    submit_btn.click(
        fn=submit_query,
        inputs=query_input,
        outputs=[output, status]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "✅ 系统就绪"),
        inputs=[],
        outputs=[query_input, output, status]
    )
    
    # 回车键提交
    query_input.submit(
        fn=submit_query,
        inputs=query_input,
        outputs=[output, status]
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