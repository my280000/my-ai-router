import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# 1. 配置路由策略（在这里定义你的“判别标准”）
# 我们可以定义一些代表“复杂任务”的关键词句
COMPLEX_TASKS = [
    "请写一段复杂的 Python 逻辑代码",
    "分析这段财务报表并给出投资建议",
    "量子物理学中纠缠态的数学推导是什么",
    "撰写一篇关于人工智能伦理的深度论文",
    "请翻译并润色这段专业的医学文献"
]

@st.cache_resource
def load_router_model():
    # 使用轻量级多语言模型，支持中文且内存占用极小
    # 模型大小约 80MB，推理速度极快
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_routing_decision(user_query, model, threshold=0.45):
    """
    计算相似度：如果用户输入与复杂任务范例的相似度超过阈值，则路由给重型模型。
    """
    # 编码向量
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    task_embeddings = model.encode(COMPLEX_TASKS, convert_to_tensor=True)
    
    # 计算余弦相似度（取范例中最高的一个）
    cosine_scores = util.cos_sim(query_embedding, task_embeddings)
    max_score = torch.max(cosine_scores).item()
    
    # 如果得分高于阈值，判定为复杂任务
    return "HEAVY" if max_score > threshold else "LIGHT", max_score

# --- UI 界面 ---
st.title("智能路由调度器 (Embedding版)")

query = st.text_input("输入你的需求：", placeholder="例如：你好 / 请帮我写个深度学习架构")

if query:
    with st.spinner('正在分析意图...'):
        model = load_router_model()
        decision, score = get_routing_decision(query, model)
    
    st.write(f"🔍 语义匹配得分: `{score:.4f}`")
    
    if decision == "HEAVY":
        st.error("🚀 路由至：**重型模型 (如 Llama 3 70B)**")
        st.caption("原因：检测到复杂逻辑、代码或专业知识需求。")
        # 这里写 requests.post 调用高端 API 的代码
    else:
        st.success("⚡ 路由至：**轻量模型 (如 Llama 3 8B / Qwen 0.5B)**")
        st.caption("原因：检测到基础对话或简单问候。")
        # 这里写调用廉价 API 的代码
