import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 预设复杂任务的特征词
# 这比单句匹配更稳，它看的是“语义密度”
COMPLEX_SAMPLES = [
    "写一段复杂的程序代码", "分析财务报表数据", "推导数学公式", 
    "撰写专业学术论文", "深度逻辑推理", "翻译长篇医学文献"
]

@st.cache_resource
def get_vectorizer():
    # 使用 TF-IDF 这种不需要深度学习框架的算法，0秒启动
    vectorizer = TfidfVectorizer()
    sample_vectors = vectorizer.fit_transform(COMPLEX_SAMPLES)
    return vectorizer, sample_vectors

st.title("智能路由调度器 (轻量版)")

query = st.text_input("输入需求：")

if query:
    vec, samples = get_vectorizer()
    query_vec = vec.transform([query])
    
    # 计算最高相似度
    similarity = cosine_similarity(query_vec, samples).max()
    
    st.write(f"📊 复杂度匹配度: `{similarity:.4f}`")
    
    # 阈值建议 0.2 - 0.3
    if similarity > 0.25:
        st.error("🚀 建议路由至：**重型模型 (70B+)**")
    else:
        st.success("⚡ 建议路由至：**轻量模型 (8B)**")
