# -*- coding: utf-8 -*-
"""
个人翻译知识库管理系统(修正版03)
- Tab1 📂 翻译项目管理:新建项目、文件上传、执行翻译(DeepSeek API).导出对照/原格式.写入历史
- Tab2 📘 术语库管理:查询/编辑/删除、CSV批量导入、统计/导出、快速搜索、批量挂接项目、历史抽取术语、分类管理
- Tab3 📊 翻译历史:查看、下载译文
- Tab4 📚 语料库管理:新增/检索/合并/Few-shot 注入
"""

import streamlit as st

from app_core.config import BASE_DIR, KBEmbedder
from app_core.database import init_db
from app_core.ui_corpus import render_corpus_manager
from app_core.ui_history import render_history_tab
from app_core.ui_index import render_index_manager_by_domain
from app_core.ui_projects import render_project_tab
from app_core.ui_terms import render_term_management

# ========== 页面设置 ==========
st.set_page_config(page_title="个人翻译知识库管理系统3.0", layout="wide")

# ========== 路径/DB ==========
conn, cur = init_db()

# ========== Session 初始化 ==========
if "kb_embedder" not in st.session_state and KBEmbedder:
    st.session_state["kb_embedder"] = KBEmbedder(lazy=True)

# ========== 页面结构 ==========
st.sidebar.title("导航")

choice = st.sidebar.radio(
    "功能选择",
    [
        "📂 翻译项目管理",
        "📘 术语库管理",
        "📊 翻译历史",
        "📚 语料库管理",
        "🧠 索引管理",
    ],
)

st.title("个人翻译知识库管理系统3.0")

# ========== Tab1:翻译项目管理 ==========
if choice.startswith("📂"):
    render_project_tab(st, cur, conn, BASE_DIR)

# ========== Tab2:术语库管理 ==========
elif choice.startswith("📘"):
    render_term_management(st, cur, conn, BASE_DIR, key_prefix="term")

# ========== Tab3:翻译历史 ==========
elif choice.startswith("📊"):
    render_history_tab(st, cur, conn)

# ========== Tab4:语料库管理 ==========
elif choice.startswith("📚"):
    render_corpus_manager(st, cur, conn)

# ========== Tab5:索引管理 ==========
elif choice.startswith("🧠"):
    render_index_manager_by_domain(st, conn, cur)
