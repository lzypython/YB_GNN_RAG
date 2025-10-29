import gradio as gr
import torch
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os
from train_gnn import GCN  # 导入 GCN 模型结构

# 配置和初始化
GRAPH_PATH_YW = "graph_yw.pkl"  # 药物评估图
MODEL_PATH_YW = "gnn_model_yw.pt"  # 药物评估模型
GRAPH_PATH_CX = "graph_cx.pkl"  # 长护险图
MODEL_PATH_CX = "gnn_model_cx.pt"  # 长护险模型
EMBED_MODEL = "/back-up/gzy/meta-comphrehensive-rag-benchmark-starter-kit/models/sentence-transformers/all-MiniLM-L6-v2/"

client = openai.OpenAI(
    api_key="empty",
    base_url="http://localhost:6081/v1"
)

# 全局变量，避免重复加载
_G_yw = None
_gnn_model_yw = None
_G_cx = None
_gnn_model_cx = None

def load_resources(task_type):
    """根据任务类型加载图和模型资源"""
    global _G_yw, _gnn_model_yw, _G_cx, _gnn_model_cx
    
    if task_type == "新增药物医保准入评估":
        if _G_yw is None or _gnn_model_yw is None:
            print("正在加载药物评估图数据...")
            with open(GRAPH_PATH_YW, "rb") as f:
                _G_yw = pickle.load(f)
            
            # 获取节点 embedding 维度
            node_emb_dim = len(_G_yw.nodes[list(_G_yw.nodes())[0]]['embedding'])
            
            # 定义模型结构并加载参数
            print("正在加载药物评估GNN模型...")
            _gnn_model_yw = GCN(in_dim=node_emb_dim, hidden_dim=64)
            _gnn_model_yw.load_state_dict(torch.load(MODEL_PATH_YW))
            print("药物评估资源加载完成！")
        
        return _G_yw, _gnn_model_yw
    
    elif task_type == "长护险失能风险预测":
        if _G_cx is None or _gnn_model_cx is None:
            print("正在加载长护险图数据...")
            with open(GRAPH_PATH_CX, "rb") as f:
                _G_cx = pickle.load(f)
            
            # 获取节点 embedding 维度
            node_emb_dim = len(_G_cx.nodes[list(_G_cx.nodes())[0]]['embedding'])
            
            # 定义模型结构并加载参数
            print("正在加载长护险GNN模型...")
            _gnn_model_cx = GCN(in_dim=node_emb_dim, hidden_dim=64)
            _gnn_model_cx.load_state_dict(torch.load(MODEL_PATH_CX))
            print("长护险资源加载完成！")
        
        return _G_cx, _gnn_model_cx
    
    else:
        raise ValueError(f"未知的任务类型: {task_type}")

from rag_inference import rag_query, rag_query_cx

# 处理用户查询的函数
def process_query(query, task_type):
    if not query.strip():
        return "请输入有效的问题。"
    
    try:
        G, gnn_model = load_resources(task_type)
        
        if task_type == "新增药物医保准入评估":
            answer = rag_query(query, G, gnn_model, topk=5)
        elif task_type == "长护险失能风险预测":
            answer = rag_query_cx(query, G, gnn_model, topk=5)
        else:
            return "请选择有效的任务类型。"
            
        return answer
        
    except Exception as e:
        return f"处理查询时出现错误：{str(e)}"

# 创建Gradio界面
with gr.Blocks(
    title="构建可信垂域基础模型，激发医保大数据要素的医疗健康活力",
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
    .task-selector {
        margin-bottom: 20px;
    }
    """
) as demo:
    
    gr.Markdown(
        """
        <div style="font-size: 1.5em; font-weight: bold;">
            🏥 构建可信垂域基础模型，激发医保大数据要素的医疗健康活力
        </div>
        
        <div class="description" style="font-size: 2.5em;">
            基于图神经网络和语言模型的智能医保评估系统，提供专业的药物准入评估和失能风险预测
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
    
    # 任务选择下拉框
    with gr.Row():
        task_type = gr.Dropdown(
            choices=["新增药物医保准入评估", "长护险失能风险预测"],
            label="🎯 选择评估任务",
            value="新增药物医保准入评估",
            info="请选择要进行的评估任务类型",
            elem_classes="task-selector"
        )
    
    with gr.Row():
        query_input = gr.Textbox(
            label="💬 请输入评估内容",
            placeholder="请根据选择的任务类型输入相应内容...",
            lines=3,
            max_lines=5,
            container=True,
            elem_classes="input-box"
        )
    
    with gr.Row():
        submit_btn = gr.Button("🚀 开始评估", variant="primary", size="lg", elem_classes="submit-btn")
        clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="lg")
    
    with gr.Row():
        output = gr.Textbox(
            label="📋 专业评估报告",
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
    
    # 示例问题 - 根据任务类型动态更新
    drug_examples = [
        ["现在有一款药，是一种血管紧张素受体脑啡肽酶抑制剂，能够显著降低心衰患者住院风险，但价格较高，它是否合适纳入医保药物名录？"],
        ["现在有一款药，治疗效果与现有药物相当但价格是现有药物的三倍，它是否合适纳入医保药物名录？"],
        ["现在有一位82岁男性，患有高血压和糖尿病，脑梗死后遗留左侧肢体偏瘫，近两年因跌倒骨折住院3次，该参保人是否属于长护险失能高风险人群？"],
        ["现在有一位65岁女性，仅有轻度高血压且控制良好，无跌倒史，日常生活完全自理，该参保人是否属于长护险失能高风险人群？"]
    ]
    
    ltc_examples = [
        ["现在有一位82岁男性，患有高血压和糖尿病，脑梗死后遗留左侧肢体偏瘫，近两年因跌倒骨折住院3次，该参保人是否属于长护险失能高风险人群？"],
        ["现在有一位65岁女性，仅有轻度高血压且控制良好，无跌倒史，日常生活完全自理，该参保人是否属于长护险失能高风险人群？"],
        ["现在有一位78岁独居女性，患有骨质疏松和重度膝关节炎，过去一年有两次跌倒记录，存在轻度认知障碍，请评估其失能风险"]
    ]
    
    examples = gr.Examples(
        examples=drug_examples,
        inputs=query_input,
        label="💡 示例问题（点击即可使用）"
    )
    
    # 底部信息
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #888; font-size: 0.9em;">
        💡 提示：本系统提供的信息仅供参考，不能替代专业医疗决策。如有医疗问题，请咨询专业医生。
        </div>
        """
    )
    
    def update_examples(task_type):
        """根据任务类型更新示例问题"""
        if task_type == "新增药物医保准入评估":
            return gr.Examples(examples=drug_examples, inputs=query_input, label="💡 示例问题（点击即可使用）")
        else:
            return gr.Examples(examples=ltc_examples, inputs=query_input, label="💡 示例问题（点击即可使用）")
    
    def update_placeholder(task_type):
        """根据任务类型更新输入框提示"""
        if task_type == "新增药物医保准入评估":
            return "例如：现在有一款药，[药物描述]，它是否合适纳入医保药物名录？"
        else:
            return "例如：现在有一位参保人，[参保人描述]，该参保人是否属于长护险失能高风险人群？"
    
    # 绑定事件
    def submit_query(query, task_type):
        if not query.strip():
            return "请输入有效的问题。", "❌ 请输入问题"
        try:
            status_msg = "⏳ 正在处理您的查询..."
            answer = process_query(query, task_type)
            return answer, "✅ 评估完成"
        except Exception as e:
            return f"处理查询时出现错误：{str(e)}", "❌ 处理错误"
    
    submit_btn.click(
        fn=submit_query,
        inputs=[query_input, task_type],
        outputs=[output, status]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "✅ 系统就绪"),
        inputs=[],
        outputs=[query_input, output, status]
    )
    
    # 任务类型变化时更新示例和提示
    task_type.change(
        fn=update_placeholder,
        inputs=task_type,
        outputs=query_input
    )
    
    # 回车键提交
    query_input.submit(
        fn=submit_query,
        inputs=[query_input, task_type],
        outputs=[output, status]
    )

if __name__ == "__main__":
    # 启动界面
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )