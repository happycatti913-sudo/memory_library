# -*- coding: utf-8 -*-
"""
个人翻译知识库管理系统(修正版03)
- Tab1 📂 翻译项目管理:新建项目、文件上传、执行翻译(DeepSeek API).导出对照/原格式.写入历史
- Tab2 📘 术语库管理:查询/编辑/删除、CSV批量导入、统计/导出、快速搜索、批量挂接项目、历史抽取术语、分类管理
- Tab3 📊 翻译历史:查看、下载译文
- Tab4 📚 语料库管理:新增/检索/合并/Few-shot 注入
"""

# ==== stdlib ====
import os
import re
import io
import sys
import json
import uuid
import time
import streamlit as st
import pandas as pd
import altair as alt

from app_core.config import (
    BASE_DIR,
    DB_PATH,
    LOG_DIR,
    LOG_FILE,
    Document,
    KBEmbedder,
    build_prompt_soft,
    build_prompt_strict,
    dedup_terms_against_db,
    highlight_terms,
    log_event,
    make_sk,
    recommend_for_segment,
    sk,
    _norm_domain_key,
    _project_domain,
)
from app_core.database import ensure_col, init_db, _get_domain_for_proj, _has_col
from app_core.projects import (
    cleanup_project_files,
    ensure_legacy_file_record,
    get_project_fewshot_enabled,
    get_project_fewshot_examples,
    get_project_ref_ids,
    fetch_project_files,
    register_project_file,
    remove_project_file,
    set_project_fewshot_enabled,
)
from app_core.semantic_index import (
    _lazy_import_vec,
    _load_index,
    _load_index_domain,
    _save_index,
    _save_index_domain,
    build_project_vector_index,
    build_strategy_index_for_domain,
    get_embedder,
    quick_diagnose_vectors,
    rebuild_project_semantic_index,
)
from app_core.semantic_ops import (
    _build_ref_context,
    align_semantic,
    semantic_consistency_report,
    semantic_retrieve,
)
from app_core.term_ops import check_term_consistency, get_terms_for_project
from app_core.translation_ops import (
    build_instruction,
    build_term_hint,
    ds_translate,
    get_deepseek,
    translate_block_with_kb,
)
from app_core.term_extraction import (
    _locate_example_pair,
    _split_sentences_for_terms,
    ds_extract_terms,
    extract_terms_with_corpus_model,
)
from app_core.file_ops import export_csv_bilingual, export_docx_bilingual, read_source_file
from app_core.corpus_ops import import_corpus_from_upload
from app_core.ui_common import render_table
from app_core.ui_index import render_index_manager, render_index_manager_by_domain
from app_core.text_utils import (
    _lazy_docx,
    _normalize,
    _split_pair_for_index,
    build_bilingual_lines,
    extract_pairs_from_docx_table,
    pair_paragraphs,
    read_docx_tables_info,
    read_docx_text,
    read_pdf_text,
    read_txt,
    split_blocks,
    split_paragraphs,
    split_sents,
)

# ========== 页面设置 ==========
st.set_page_config(page_title="个人翻译知识库管理系统3.0", layout="wide")

# ========== 工具函数 ==========
# ======= 获取某条历史记录对应的原文(优先 items.body.兜底 src_path 仅作为标题提示)=======
def run_project_translation_ui(
    pid,
    project_title,
    src_path,
    conn,
    cur
):
    """
    执行翻译整个 UI + 逻辑。不改逻辑，只是把原来 Tab1 里的大块搬进来。
    参数含义：
        pid: 当前项目 ID
        project_title: 项目标题
        src_path: 源文件路径或文本内容
        conn, cur: 数据库句柄
    """

    st.subheader(f"📘 项目：{project_title}")
    st.info("打工不易，牛马哭泣。")

    # 先统一加载术语（静态 + 动态）
    term_map, term_meta = get_terms_for_project(cur, pid, use_dynamic=True)
    proj_terms_all = term_map  # 给 _detect_hits 用

    # 1) 结果缓存初始化(统一用 session_state)
    if "all_results" not in st.session_state:
        st.session_state["all_results"] = []
    st.session_state["all_results"].clear()

    # 2) 环境检查
    ak, model = get_deepseek()
    if not ak:
        st.error("未检测到 DeepSeek Key.请在 `.streamlit/secrets.toml` 配置 [deepseek]")
        st.stop()
    if not selected_src_path or not os.path.exists(selected_src_path):
        st.error("缺少源文件")
        st.stop()

    # 3) 读取与分段
    src_text = read_source_file(selected_src_path)
    st.code(repr(src_text[:400]))                     # 看字符串里有没有 '\n'
    st.write({"len": len(src_text), "nl": src_text.count("\n"), "cr": src_text.count("\r")})
    st.write({"preview_lines": src_text.splitlines()[:3]})
    
    # 用统一的 split_paragraphs 做切分
    blocks = split_paragraphs(src_text)
    if not blocks:
        st.error("源文件内容为空，或未识别到有效段落")
        st.stop()

    st.info(f"按段落切分，共 {len(blocks)} 段，开始翻译…")

    lang_pair_val = st.session_state.get(f"lang_{pid}", "中译英")
    use_semantic  = bool(st.session_state.get(f"use_sem_{pid}", True))
    scope_val     = st.session_state.get(f"scope_{pid}", st.session_state.get("scope_newproj", "project"))

    # 4) 循环翻译（统一走 translate_block_with_kb 管线）
    # 先加载 few-shot 示例（项目级）
    fewshot_examples = get_project_fewshot_examples(cur, pid, limit=5)
    if fewshot_examples:
        with st.expander("📌 Few-shot 参考示例(项目级注入)", expanded=False):
            for ex in fewshot_examples:
                st.markdown(
                    f"**{ex['title']}**\n\n源文:\n{ex['src']}\n\n译文:\n{ex['tgt']}\n---"
                )

    # 为每次翻译准备一个结果列表
    if "all_results" not in st.session_state:
        st.session_state["all_results"] = []
    st.session_state["all_results"].clear()

    # DeepSeek key / model 只取一次
    ak, model = get_deepseek()
    if not ak:
        st.error("未检测到 DeepSeek Key.请在 `.streamlit/secrets.toml` 配置 [deepseek]")
        st.stop()

    # 按段循环翻译
    for i, blk in enumerate(blocks, start=1):
        blk = str(blk or "").strip()
        if not blk:
            continue

        # —— 调用统一管线完成“术语 + 参考 + DeepSeek” —— 
        out_text = res["tgt"]
        term_map_all = res["term_map_all"]
        terms_in_block = res.get("terms_in_block", {})
        terms_corpus = res.get("terms_corpus_dyn", {})
        terms_final = res.get("terms_final", {})
        ref_context = res["ref_context"]
        violated = res["violated_terms"]

        # —— 展示术语 + 参考例句（折叠，可选）——
        with st.expander(f"第 {i} 段 · 术语与参考（可选展开）", expanded=False):
            if term_map_all:
                df_all = pd.DataFrame(
                    [
                        {
                            "术语": s,
                            "译文": t,
                            "命中本段": s in terms_in_block,
                            "命中参考例句": s in terms_corpus,
                            "最终注入Prompt": s in terms_final,
                        }
                        for s, t in term_map_all.items()
                    ]
                )
                st.dataframe(df_all, width='stretch')

                # 单独列出“语料驱动术语”（方便你看）
                corpus_only = [
                    {"术语": s, "译文": t}
                    for s, t in terms_corpus.items()
                ]
                if corpus_only:
                    st.markdown("**语料驱动术语（仅在参考例句中命中的术语）：**")
                    st.dataframe(pd.DataFrame(corpus_only), width='stretch')

            if ref_context:
                st.text(ref_context[:1500])

        # 记录结果(统一用 session_state)
        st.session_state["all_results"].append(out_text)
        st.write(f"✅ 第 {i} 段完成")

        # 译后一致性提醒
        if violated:
            st.warning("以下术语未在译文中出现(建议人工核对): " + "；".join(violated))


    # ===== 循环结束后:汇总输出 =====

    # 结果与源段落兜底与对齐
    all_results_safe = list(st.session_state.get("all_results", []))
    blocks_src_safe  = list(blocks if 'blocks' in locals() else [])
    if len(blocks_src_safe) != len(all_results_safe):
        n = min(len(blocks_src_safe), len(all_results_safe))
        blocks_src_safe  = blocks_src_safe[:n]
        all_results_safe = all_results_safe[:n]

    # 文本汇总
    final_text = "\n\n".join(all_results_safe)

    # 一致性报告(语义+术语)
    try:
        term_map_report, _ = get_terms_for_project(cur, pid, use_dynamic=True)
        df_rep = semantic_consistency_report(
            project_id=pid,
            blocks_src=blocks_src_safe,
            blocks_tgt=all_results_safe,
            term_map=term_map_report,
            topk=3,
            thr=0.70,
            cur=cur,
        )
        st.markdown("### 🔎 译后一致性报告(语义+术语)")
        st.dataframe(df_rep, width='stretch')
    except Exception as e:
        st.caption(f"(一致性报告生成失败: {e})")

    # 下载按钮(TXT)
    proj_title = (title if 'title' in locals() and title else f"project_{pid}")
    st.download_button(
        "⬇️ 下载翻译结果 (TXT)",
        final_text or "",
        file_name=f"{proj_title}_翻译结果.txt",
        mime="text/plain",
        key=f"dl_txt_{pid}"
    )

    # 写入历史 trans_ext
    try:
        # 兼容取值(都做了兜底.防止未定义)
        src_path = selected_src_path if ('selected_src_path' in locals() and selected_src_path) else None
        mode_val = mode if ('mode' in locals() and mode) else "标准模式"
        lang_pair_val = st.session_state.get(f"lang_{pid}", "中译英")

        # 计算段数:建议以「源文本」统计；若无则退回最终译文
        src_for_seg = src_text if ('src_text' in locals() and src_text) else final_text
        seg_count = len(split_sents(src_for_seg, lang_hint="auto"))

        cur.execute("""
            INSERT INTO trans_ext (
                project_id, src_path, lang_pair, mode, output_text,
                stats_json, segments, term_hit_total, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            pid,             # 项目ID
            src_path,        # 源文件路径(可为空)
            lang_pair_val,   # 语对
            mode_val,        # 模式
            final_text,      # 输出文本(最终译文)
            None,            # 统计JSON(占位)
            seg_count,       # 段数(修复:不用 blocks_src_safe)
            None             # 术语命中数(占位.可后续填真实值)
        ))
        conn.commit()
        st.success("📝 已写入翻译历史")
    except Exception as e:
        st.warning(f"写入翻译历史失败: {e}")
            

# ======= 对齐并导出(依赖你已有的 split_blocks / align_export)=======
# ========== 路径/DB ==========
conn, cur = init_db()

# ====== 术语管理 UI ======

def render_term_management(st, cur, conn, base_dir, key_prefix="term"):
    sk = make_sk(key_prefix)

    st.subheader("📘 术语库管理")
    sub_tabs = st.tabs(["查询与编辑", "批量导入 CSV", "统计与导出", "快速搜索", "批量挂接项目", "从历史提取术语", "分类管理"])

    # —— 查询与编辑
    with sub_tabs[0]:
        sk0 = lambda name: f"{key_prefix}_t0_{name}"

        c1, c2, c3, c4 = st.columns(4)
        with c1: kw = st.text_input("关键词(源/目标/例句)", "", key=sk("kw_example"))
        with c2: dom = st.text_input("领域", "", key=sk("dom"))
        with c3: strat = st.text_input("策略", "", key=sk("strat"))
        with c4: pid = st.text_input("项目ID过滤", "", key=sk("pid"))
        cat = st.text_input("分类(支持子串)", "", key=sk("cat"))

        # —— 兼容老库:如无 category 列则补列(已有会忽略)
        try:
            cur.execute("ALTER TABLE term_ext ADD COLUMN category TEXT;")
            conn.commit()
        except Exception:
            pass

        # 1) 以数据库真实列为准检测是否存在 category
        cols_db = [c[1].lower() for c in cur.execute("PRAGMA table_info(term_ext);").fetchall()]
        has_category = ("category" in cols_db)

        # 2) 拼 SQL(只在 DB 真的有该列时才 SELECT category)
        base_cols = "id, source_term, target_term, domain, project_id, strategy, example"
        sel_cols = base_cols + (", category" if has_category else "")
        sql = f"SELECT {sel_cols} FROM term_ext WHERE 1=1"
        params = []

        if kw:
            like = f"%{kw}%"
            sql += " AND (IFNULL(source_term,'') LIKE ? OR IFNULL(target_term,'') LIKE ? OR IFNULL(example,'') LIKE ?)"
            params += [like, like, like]

        if dom:
            sql += " AND IFNULL(domain,'') LIKE ?"
            params += [f"%{dom}%"]

        if strat:
            sql += " AND IFNULL(strategy,'') LIKE ?"
            params += [f"%{strat}%"]

        if pid and str(pid).isdigit():
            sql += " AND IFNULL(project_id,0) = ?"
            params += [int(pid)]

        if has_category and cat:
            sql += " AND IFNULL(category,'') LIKE ?"
            params += [f"%{cat}%"]

        sql += " ORDER BY source_term COLLATE NOCASE LIMIT 1000"

        # 3) 查询并构造 DataFrame(表头与实际列对齐)
        rows = cur.execute(sql, params).fetchall()
        headers = ["ID","源术语","目标术语","领域","项目ID","策略","例句"]
        if has_category:
            headers.append("分类")

        df = pd.DataFrame(rows, columns=headers)
        st.caption(f"当前查询返回:{len(df)} 条")

        # 4) 空数据就不渲染编辑器
        if df.empty:
            st.info("没有匹配的术语。")
        else:
            # === 用 session_state 维护“当前表格”，含选择列 ===
            editor_df_key = sk0("editor_df")   # 专门存 DataFrame
            editor_key    = sk0("editor")      # data_editor 小部件本身

            # 初始化 / 尺寸变化时重置
            if editor_df_key not in st.session_state:
                work_df = df.copy()
                if "sel" not in work_df.columns:
                    work_df.insert(0, "sel", False)
                st.session_state[editor_df_key] = work_df
            else:
                work_df = st.session_state[editor_df_key]
                if len(work_df) != len(df):
                    work_df = df.copy()
                    if "sel" not in work_df.columns:
                        work_df.insert(0, "sel", False)
                    st.session_state[editor_df_key] = work_df

            # 动态构建列配置.只有当 DB 真有“分类”时才加入
            col_cfg = {
                "ID": st.column_config.NumberColumn("ID", disabled=True),
                "sel": st.column_config.CheckboxColumn("选择"),
                "源术语": "源术语",
                "目标术语": "目标术语",
                "领域": "领域",
                "项目ID": st.column_config.NumberColumn("项目ID", step=1, required=False),
                "策略": "策略",
                "例句": st.column_config.TextColumn("例句"),
            }
            if has_category:
                col_cfg["分类"] = st.column_config.TextColumn("分类")

            # 真正的编辑器:以 session_state 里的 DataFrame 为准
            edited = st.data_editor(
                st.session_state[editor_df_key],
                num_rows="dynamic",
                key=editor_key,
                column_config=col_cfg,
            )
            # 把用户这次编辑结果写回 session_state
            st.session_state[editor_df_key] = edited

            c1, c2, c3 = st.columns([1, 1, 2])

            # ---------------- 保存修改 ----------------
            with c1:
                if st.button("💾 保存修改", type="primary", key=sk("save_terms")):
                    updated = inserted = 0
                    for _, row in edited.iterrows():
                        if pd.notna(row["ID"]):  # 更新
                            if has_category:
                                cur.execute("""
                                    UPDATE term_ext
                                    SET source_term=?, target_term=?, domain=?, project_id=?, strategy=?, example=?, category=?
                                    WHERE id=?;
                                """, (
                                    (row["源术语"] or "").strip(),
                                    (row["目标术语"] or None),
                                    (row["领域"] or None),
                                    (int(row["项目ID"]) if pd.notna(row["项目ID"]) else None),
                                    (row["策略"] or None),
                                    (row["例句"] or None),
                                    (row.get("分类") or None),
                                    int(row["ID"])
                                ))
                            else:
                                cur.execute("""
                                    UPDATE term_ext
                                    SET source_term=?, target_term=?, domain=?, project_id=?, strategy=?, example=?
                                    WHERE id=?;
                                """, (
                                    (row["源术语"] or "").strip(),
                                    (row["目标术语"] or None),
                                    (row["领域"] or None),
                                    (int(row["项目ID"]) if pd.notna(row["项目ID"]) else None),
                                    (row["策略"] or None),
                                    (row["例句"] or None),
                                    int(row["ID"])
                                ))
                            updated += cur.rowcount
                        else:  # 新增
                            if str(row["源术语"]).strip():
                                if has_category:
                                    cur.execute("""
                                        INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example, category)
                                        VALUES (?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        (row["源术语"] or "").strip(),
                                        (row["目标术语"] or None),
                                        (row["领域"] or None),
                                        (int(row["项目ID"]) if pd.notna(row["项目ID"]) else None),
                                        (row["策略"] or None),
                                        (row["例句"] or None),
                                        (row.get("分类") or None)
                                    ))
                                else:
                                    cur.execute("""
                                        INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example)
                                        VALUES (?, ?, ?, ?, ?, ?)
                                    """, (
                                        (row["源术语"] or "").strip(),
                                        (row["目标术语"] or None),
                                        (row["领域"] or None),
                                        (int(row["项目ID"]) if pd.notna(row["项目ID"]) else None),
                                        (row["策略"] or None),
                                        (row["例句"] or None),
                                    ))
                                inserted += 1
                    conn.commit()
                    st.success(f"✅ 已保存:更新 {updated} 条.新增 {inserted} 条。")
                    st.rerun()

            # ---------------- 全选 / 清空 / 删除 ----------------
            with c2:
                cc2a, cc2b, cc2c = st.columns([1, 1, 2])
                # 这里统一操作 session_state 里的 DataFrame
                cur_df = st.session_state[editor_df_key]

                if cc2a.button("全选", key=sk("sel_all")):
                    cur_df.loc[:, "sel"] = True
                    st.session_state[editor_df_key] = cur_df
                    st.rerun()

                if cc2b.button("清空", key=sk("sel_clear")):
                    cur_df.loc[:, "sel"] = False
                    st.session_state[editor_df_key] = cur_df
                    st.rerun()

                if cc2c.button("🗑️ 删除已勾选", key=sk("del_sel")):
                    to_delete = cur_df[(cur_df["sel"] == True) & pd.notna(cur_df["ID"])]["ID"].astype(int).tolist()
                    if not to_delete:
                        st.warning("未勾选任何记录")
                    else:
                        cur.executemany("DELETE FROM term_ext WHERE id=?", [(i,) for i in to_delete])
                        conn.commit()
                        st.success(f"🗑️ 已删除 {len(to_delete)} 条")
                        st.rerun()

                with c3:
                    proj_opts = cur.execute(
                        "SELECT id, title FROM items WHERE COALESCE(type,'')='project' ORDER BY id DESC"
                    ).fetchall()
                    proj_map = {"(不挂接/置空)": None, **{f"#{i} {t}": i for (i, t) in proj_opts}}

                    cc3a, cc3b = st.columns([2, 1])
                    target_proj_label = cc3a.selectbox("批量挂接到项目", list(proj_map.keys()), key=sk("bind_proj_sel"))
                    if cc3b.button("执行挂接", type="primary", key=sk("bind_apply")):
                        if "ID" in edited.columns:
                            to_update = edited[(edited["sel"] == True) & pd.notna(edited["ID"])]["ID"].astype(int).tolist()
                        else:
                            to_update = []
                        if not to_update:
                            st.warning("未勾选任何记录")
                        else:
                            pid_val = proj_map.get(target_proj_label)
                            q_marks = ",".join("?" for _ in to_update)
                            cur.execute(f"UPDATE term_ext SET project_id=? WHERE id IN ({q_marks})", (pid_val, *to_update))
                            conn.commit()
                            st.success(f"✅ 已挂接 {len(to_update)} 条到项目:{target_proj_label or '(空)'}")
                            st.rerun()

            st.markdown("### 单条新增 / 编辑")
            with st.form(sk0("term_edit")):
                col1, col2, col3 = st.columns(3)
                with col1:
                    rid_edit = st.text_input("要编辑的记录 ID(留空则新增)", "", key=sk("rid_edit"))
                    source_term = st.text_input("源语言术语(必填)*", key=sk("source_term"))
                    target_term = st.text_input("目标语言术语", key=sk("target_term"))
                with col2:
                    domain = st.text_input("领域", key=sk("domain"))
                    project_id = st.text_input("项目ID(可空)", key=sk("project_id"))
                    strategy = st.text_input("策略(直译/意译/转译/音译/省略/增译/规范化…)", key=sk("strategy"))
                with col3:
                    example = st.text_area("例句", height=80, key=sk("example"))
                    category = st.text_input("分类(可选)", key=sk("category"))

                b1, b2 = st.columns(2)
                add = b1.form_submit_button("保存(新增或更新)")
                delbtn = b2.form_submit_button("删除(按 ID)")

            if add:
                if not source_term.strip():
                    st.error("源术语必填")
                else:
                    if rid_edit and rid_edit.isdigit():
                        if _has_col("term_ext", "category"):
                            cur.execute("""
                            UPDATE term_ext
                            SET source_term=?, target_term=?, domain=?, project_id=?, strategy=?, example=?, category=?
                            WHERE id=?;
                            """, (source_term.strip(), target_term.strip() or None, domain or None,
                                int(project_id) if project_id.isdigit() else None, strategy or None, example or None,
                                category or None, int(rid_edit)))
                        else:
                            cur.execute("""
                            UPDATE term_ext
                            SET source_term=?, target_term=?, domain=?, project_id=?, strategy=?, example=?
                            WHERE id=?;
                            """, (source_term.strip(), target_term.strip() or None, domain or None,
                                int(project_id) if project_id.isdigit() else None, strategy or None, example or None,
                                int(rid_edit)))
                        conn.commit(); st.success("✅ 已更新"); st.rerun()
                    else:
                        if _has_col("term_ext", "category"):
                            cur.execute("""
                            INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example, category)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, (
                                source_term.strip(),
                                (target_term.strip() or None) if target_term else None,
                                (domain or None),
                                (int(project_id) if project_id.isdigit() else None) if project_id else None,
                                (strategy or None),
                                (example or None),
                                (category or None)
                            ))
                        else:
                            cur.execute("""
                            INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                source_term.strip(),
                                (target_term.strip() or None) if target_term else None,
                                (domain or None),
                                (int(project_id) if project_id.isdigit() else None) if project_id else None,
                                (strategy or None),
                                (example or None),
                            ))
                        conn.commit(); st.success("✅ 已新增"); st.rerun()

            if delbtn:
                if rid_edit and rid_edit.isdigit():
                    cur.execute("DELETE FROM term_ext WHERE id=?", (int(rid_edit),))
                    conn.commit(); st.success("🗑️ 已删除"); st.rerun()
                else:
                    st.error("请填写要删除的 ID")

    # —— 批量导入 CSV(增强版:列名规范化 / 动态带或不带 category / 去重或Upsert / 逐行容错)
    with sub_tabs[1]:
        sk1 = lambda n: f"{key_prefix}_t1_{n}"
        st.caption("CSV 推荐列:source_term, target_term, domain, project_id, strategy, example(可选:category / 分类)")
        up = st.file_uploader("上传 CSV 文件", type=["csv"], key=sk1("csv"))

        # 小工具:补列 + 唯一索引 + 规范函数
        def _ensure_col(table, col, type_):
            try:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {type_};")
                conn.commit()
            except Exception:
                pass

        def _ensure_unique_index():
            try:
                cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_term_proj
                ON term_ext(LOWER(TRIM(source_term)), IFNULL(project_id,-1));
                """)
                conn.commit()
            except Exception:
                pass

        def _norm_cols(df):
            # 列名:去BOM/两端空格 -> 小写 -> 替换空格为下划线 -> 中英列名映射
            df.columns = [str(c).replace("\ufeff","").strip() for c in df.columns]
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            mapping = {
                "源术语": "source_term", "source": "source_term",
                "目标术语": "target_term", "target": "target_term",
                "领域": "domain",
                "项目id": "project_id", "项目_id": "project_id",
                "策略": "strategy",
                "例句": "example",
                "分类": "category",
            }
            df = df.rename(columns={k.lower(): v for k, v in mapping.items()})
            return df

        def _norm(v):
            if v is None: return None
            v = str(v).strip().replace("\u3000", " ")
            return v if v else None

        def _to_int(v):
            try:
                return int(v) if v is not None and str(v).strip() != "" else None
            except Exception:
                return None

        if up is not None:
            try:
                df_up = pd.read_csv(up, encoding="utf-8-sig")
            except Exception:
                df_up = pd.read_csv(up, encoding="utf-8", errors="ignore")

            df_up = _norm_cols(df_up)
            st.write("检测到列:", list(df_up.columns))
            render_table(df_up.head(10), key=sk("csv_preview"))

            # DB 侧确保有 category 列(兼容老库;若已有会忽略)
            _ensure_col("term_ext", "category", "TEXT")

            # 侧边选项
            c1, c2, c3 = st.columns(3)
            with c1:
                dedup = st.checkbox("去重(源术语+项目ID)", value=True, key=sk1("dedup"))
            with c2:
                use_upsert = st.checkbox("已存在则更新(Upsert)", value=False, key=sk1("upsert"))
            with c3:
                skip_empty = st.checkbox("跳过空译文", value=False, key=sk1("skip_empty"))

            # Upsert 需要唯一索引
            if use_upsert:
                _ensure_unique_index()

            if st.button("导入术语库", key=sk1("import_btn")):
                # 是否包含 category 列(以CSV为准;DB已有不强制CSV必须有)
                has_category_col = ("category" in df_up.columns)

                # 去重缓存(仅在非Upsert模式下使用)
                existing = set()
                if dedup and not use_upsert:
                    rows_exist = cur.execute("""
                        SELECT LOWER(TRIM(source_term)), IFNULL(project_id,-1)
                        FROM term_ext
                    """).fetchall()
                    existing = set(rows_exist)

                ins = skp = upd = 0
                errors = []

                for idx, row in df_up.iterrows():
                    src = _norm(row.get("source_term"))
                    if not src:
                        skp += 1
                        continue

                    tgt = _norm(row.get("target_term"))
                    if skip_empty and not tgt:
                        skp += 1
                        continue

                    dom = _norm(row.get("domain"))
                    pid = _to_int(row.get("project_id"))
                    stg = _norm(row.get("strategy"))
                    exa = _norm(row.get("example"))
                    cat = _norm(row.get("category")) if has_category_col else None

                    try:
                        if use_upsert:
                            # Upsert 分支(需要唯一索引)
                            if has_category_col:
                                cur.execute("""
                                INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example, category)
                                VALUES (?,?,?,?,?,?,?)
                                ON CONFLICT(LOWER(TRIM(source_term)), IFNULL(project_id,-1))
                                DO UPDATE SET
                                    target_term=COALESCE(excluded.target_term, term_ext.target_term),
                                    domain     =COALESCE(excluded.domain,      term_ext.domain),
                                    strategy   =COALESCE(excluded.strategy,    term_ext.strategy),
                                    example    =COALESCE(excluded.example,     term_ext.example),
                                    category   =COALESCE(excluded.category,    term_ext.category);
                                """, (src, tgt, dom, pid, stg, exa, cat))
                            else:
                                cur.execute("""
                                INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example)
                                VALUES (?,?,?,?,?,?)
                                ON CONFLICT(LOWER(TRIM(source_term)), IFNULL(project_id,-1))
                                DO UPDATE SET
                                    target_term=COALESCE(excluded.target_term, term_ext.target_term),
                                    domain     =COALESCE(excluded.domain,      term_ext.domain),
                                    strategy   =COALESCE(excluded.strategy,    term_ext.strategy),
                                    example    =COALESCE(excluded.example,     term_ext.example);
                                """, (src, tgt, dom, pid, stg, exa))
                            upd += 1  # 计为“处理成功”.不区分新旧
                        else:
                            # 非 Upsert:去重(源术语+项目ID)
                            key = (src.lower(), pid if pid is not None else -1)
                            if dedup and key in existing:
                                skp += 1
                                continue

                            if has_category_col:
                                cur.execute("""
                                    INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example, category)
                                    VALUES (?,?,?,?,?,?,?)
                                """, (src, tgt, dom, pid, stg, exa, cat))
                            else:
                                cur.execute("""
                                    INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example)
                                    VALUES (?,?,?,?,?,?)
                                """, (src, tgt, dom, pid, stg, exa))
                            ins += 1
                            if dedup:
                                existing.add(key)

                    except Exception as e:
                        errors.append((idx+1, src, str(e)))
                        skp += 1
                        continue

                conn.commit()

                # 结果提示
                if use_upsert:
                    st.success(f"✅ 已处理 {ins+upd} 条(其中可能含新增+更新).跳过 {skp} 条。")
                else:
                    st.success(f"✅ 新增 {ins} 条.跳过 {skp} 条。")

                if errors:
                    with st.expander("❗ 行级错误明细(不影响其他行写入)", expanded=False):
                        for i, s, e in errors:
                            st.write(f"第 {i} 行({s}):{e}")


    # —— 统计与导出
    with sub_tabs[2]:
        sk2 = lambda n: f"{key_prefix}_t2_{n}"
        st.markdown("#### 术语统计")
        df_stats = pd.read_sql_query("SELECT strategy, domain, category FROM term_ext WHERE source_term IS NOT NULL", conn)
        if df_stats.empty:
            st.info("术语库为空.请先添加或导入")
        else:
            # 统一预处理：空值 → 未标注
            df_stats["strategy"] = df_stats["strategy"].fillna("未标注").replace("", "未标注")
            df_stats["domain"]   = df_stats["domain"].fillna("未标注").replace("", "未标注")

            # 选择统计维度
            dim_label = st.selectbox(
                "统计维度",
                ["按领域 (domain)", "按翻译策略 (strategy)"],
                index=0,
                key=sk2("dim_sel"),
            )

            if "领域" in dim_label:
                dim_col = "domain"
                dim_title = "领域"
            else:
                dim_col = "strategy"
                dim_title = "翻译策略"

            # 选择展示方式
            chart_type = st.radio(
                "展示方式",
                ["柱状图", "饼图", "数据表"],
                index=0,
                horizontal=True,
                key=sk2("chart_type"),
            )

            # 做计数
            count_df = (
                df_stats.groupby(dim_col)[dim_col]
                .count()
                .reset_index(name="term_count")
                .sort_values("term_count", ascending=False)
            )

            # ===== 不同展示方式 =====
            if chart_type == "柱状图":
                st.markdown(f"**{dim_title} 分布（柱状图）**")

                chart = (
                    alt.Chart(count_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("term_count:Q", title="术语数量"),
                        y=alt.Y(f"{dim_col}:N", sort="-x", title=dim_title),
                        tooltip=[dim_col, "term_count"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, width='stretch')

            elif chart_type == "饼图":
                st.markdown(f"**{dim_title} 分布（饼图）**")

                chart = (
                    alt.Chart(count_df)
                    .mark_arc()
                    .encode(
                        theta=alt.Theta("term_count:Q", title="术语数量"),
                        color=alt.Color(f"{dim_col}:N", title=dim_title),
                        tooltip=[dim_col, "term_count"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, width='stretch')

            else:  # 数据表
                st.markdown(f"**{dim_title} 分布（数据表）**")
                tbl = count_df.rename(
                    columns={
                        dim_col: dim_title,
                        "term_count": "术语数量",
                    }
                )
                render_table(tbl, hide_index=True, key=sk2("tbl"))

        st.markdown("---")
        st.markdown("#### 导出术语表")
        df_exp = pd.read_sql_query("""
            SELECT source_term AS '源术语',
                   target_term AS '目标术语',
                   domain AS '领域',
                   strategy AS '翻译策略',
                   example AS '示例句',
                   category AS '分类'
            FROM term_ext ORDER BY source_term COLLATE NOCASE
        """, conn)

        from io import BytesIO
        buff = BytesIO()
        with pd.ExcelWriter(buff, engine="xlsxwriter") as writer:
            df_exp.to_excel(writer, index=False, sheet_name="术语库")
        st.download_button("📥 下载 Excel",
                           buff.getvalue(),
                           file_name="terms.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key=sk2("dl_terms"))

    # —— 快速搜索
    with sub_tabs[3]:
        sk3 = lambda n: f"{key_prefix}_t3_{n}"
        q = st.text_input("快速搜索(前缀/子串)", "", key=sk3("q"))
        limit = st.number_input("返回上限", 1, 5000, 1000, 100, key=sk3("limit"))
        if st.button("搜索", key=sk3("q_btn")):
            if q:
                like = f"%{q}%"
                rows = cur.execute("""
                    SELECT id, source_term, target_term, domain, project_id
                    FROM term_ext
                    WHERE source_term LIKE ? OR target_term LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (like, like, int(limit))).fetchall()
                render_table(pd.DataFrame(rows, columns=["ID","源术语","目标术语","领域","项目"]),
                             key=sk3("q_grid"),editable=True)
            else:
                st.warning("请输入关键词")

    # —— 批量挂接项目
    with sub_tabs[4]:
        sk4 = lambda n: f"{key_prefix}_t4_{n}"
        st.caption("将一批术语统一设置 project_id.便于项目内优先匹配")
        ids_txt = st.text_area("术语ID列表(逗号/空格/换行分隔)", key=sk4("ids"))
        pid_to = st.text_input("目标项目ID", key=sk4("pid_to"))
        if st.button("批量挂接", key=sk4("batch_btn")):
            import re
            if not pid_to.isdigit():
                st.error("项目ID需为数字")
            else:
                raw = re.split(r"[,\s]+", ids_txt.strip())
                ids = [int(x) for x in raw if x.isdigit()]
                if not ids:
                    st.warning("未识别到有效ID")
                else:
                    qmarks = ",".join(["?"]*len(ids))
                    cur.execute(f"UPDATE term_ext SET project_id=? WHERE id IN ({qmarks})", (int(pid_to), *ids))
                    conn.commit()
                    st.success(f"✅ 已挂接 {len(ids)} 条到项目 {pid_to}")

    # —— 从历史提取术语
    with sub_tabs[5]:
        sk5 = lambda n: f"{key_prefix}_t5_{n}"
        st.markdown("#### 从翻译历史记录抽取术语(DeepSeek)")
        ak, model = get_deepseek()
        if not ak:
            st.info("未检测到 DeepSeek Key，将直接使用语料库同款模型做结构化术语抽取。")

        mode_pick = st.radio(
            "选择来源",
            ["按项目抽取(合并多条)", "按单条记录抽取"],
            horizontal=True,
            key=sk5("ext_mode"),
        )
        if mode_pick == "按项目抽取(合并多条)":
            pid_ext = st.text_input("项目ID", key=sk5("ext_pid"))
            max_chars = st.number_input("采样最大字符数", 1000, 20000, 5000, 500, key=sk5("ext_max"))
            if st.button("开始抽取", key=sk5("ext_go_proj")):
                if pid_ext.isdigit():
                    rows = cur.execute(
                        "SELECT src_path, output_text FROM trans_ext WHERE project_id=? ORDER BY id DESC LIMIT 10",
                        (int(pid_ext),),
                    ).fetchall()
                    buf = []
                    total = 0
                    for sp, ot in rows:
                        src = read_source_file(sp) if sp else ""
                        txt = (src + "\n" + (ot or "")).strip()
                        if not txt:
                            continue
                        if total + len(txt) > int(max_chars):
                            remain = max(0, int(max_chars) - total)
                            buf.append(txt[:remain])
                            break
                        else:
                            buf.append(txt)
                            total += len(txt)
                    big = "\n\n".join(buf)
                    big = "\n\n".join(buf)

                    # ✅ 先看这个项目到底有没有可用的历史文本
                    if not big.strip():
                        st.warning("该项目下没有可用的翻译历史文本，无法抽取术语。")
                        return

                    # 调试用:你可以先看看采样了多少字、前几行是什么
                    st.write({
                        "history_rows": len(rows),
                        "sample_chars": len(big),
                        "sample_preview": big[:300]
                    })

                    try:
                        res = ds_extract_terms(big, ak, model, src_lang="zh", tgt_lang="en", prefer_corpus_model=True)
                    except Exception as e:
                        st.error(f"调用术语抽取时出错: {e}")
                        return

                    # 调试用:先看一下原始结果长什么样
                    st.write({"extract_result_preview": str(res)[:500]})
                    if not res:
                        st.info("未抽取到术语或解析失败")
                    else:
                        st.success(f"抽取到 {len(res)} 条.准备批量写入……")
                        ins = 0
                        for o in res:
                            cur.execute(
                                "INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example) VALUES (?, ?, ?, ?, ?, ?)",
                                (o["source_term"], o.get("target_term") or None, o.get("domain"), int(pid_ext), o.get("strategy"), o.get("example")),
                            )
                            ins += 1
                        conn.commit()
                        st.success(f"✅ 已写入术语库 {ins} 条")
                else:
                    st.warning("请输入数字型项目ID")
        else:
            rid_ext = st.text_input("历史记录ID", key=sk5("ext_rid"))
            if st.button("开始抽取", key=sk5("ext_go_rec")):
                if rid_ext.isdigit():
                    row = cur.execute(
                        "SELECT src_path, output_text, project_id FROM trans_ext WHERE id=?",
                        (int(rid_ext),),
                    ).fetchone()
                    if not row:
                        st.warning("未找到该记录")
                    else:
                        sp, ot, pid0 = row
                        src = read_source_file(sp) if sp else ""
                        big = (src + "\n" + (ot or "")).strip()
                        res = ds_extract_terms(big, ak, model, src_lang="zh", tgt_lang="en", prefer_corpus_model=True)
                        if not res:
                            st.info("未抽取到术语或解析失败")
                        else:
                            st.success(f"抽取到 {len(res)} 条.准备批量写入……")
                            ins = 0
                            for o in res:
                                cur.execute(
                                    "INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example) VALUES (?, ?, ?, ?, ?, ?)",
                                    (o["source_term"], o.get("target_term") or None, o.get("domain"), pid0, o.get("strategy"), o.get("example")),
                                )
                                ins += 1
                            conn.commit()
                            st.success(f"✅ 已写入术语库 {ins} 条(project_id={pid0})")

    # —— 分类管理
    with sub_tabs[6]:
        sk6 = lambda n: f"{key_prefix}_t6_{n}"
        st.markdown("#### 分类管理")
        c1, c2 = st.columns(2)
        with c1:
            ids_txt = st.text_area("按 ID 批量设置分类(逗号/空格/换行分隔)", key=sk6("cat_ids"))
            cat_to = st.text_input("要设置的分类名", key=sk6("cat_name"))
            if st.button("批量设置分类", key=sk6("cat_set_ids")):
                import re
                raw = re.split(r"[,\s]+", (ids_txt or "").strip())
                ids = [int(x) for x in raw if x.isdigit()]
                if not ids or not cat_to.strip():
                    st.warning("请填入ID列表与分类名称")
                else:
                    qmarks = ",".join(["?"] * len(ids))
                    cur.execute(f"UPDATE term_ext SET category=? WHERE id IN ({qmarks})", (cat_to.strip(), *ids))
                    conn.commit()
                    st.success(f"✅ 已设置 {len(ids)} 条为分类:{cat_to.strip()}")

        with c2:
            pid_cat = st.text_input("将某项目ID全部术语设置为分类", key=sk6("cat_pid"))
            cat2_to = st.text_input("分类名", key=sk6("cat2_name"))
            if st.button("按项目ID统一分类", key=sk6("cat_set_pid")):
                if pid_cat.isdigit() and cat2_to.strip():
                    cur.execute("UPDATE term_ext SET category=? WHERE project_id=?", (cat2_to.strip(), int(pid_cat)))
                    conn.commit()
                    st.success(f"✅ 已将项目 {pid_cat} 的术语分类设为:{cat2_to.strip()}")
                else:
                    st.warning("请填写项目ID与分类名")

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
# ========== Tab1:翻译项目管理 ==========
if choice.startswith("📂"):
    st.subheader("翻译项目管理")
    with st.form("new_project"):
        TAG_OPTIONS = ["政治", "经济", "文化", "文物", "金融", "法律"]
        SCENE_OPTIONS = ["学术", "配音稿", "正式会议"]
        use_semantic = st.checkbox("在翻译时启用语义召回参考", value=True)
        
        # === 语义召回范围选择 ===
        scope_label = "语义召回范围"
        scope_options = {
            "仅当前项目": "project",
            "同领域 + 当前项目": "domain",
            "全库": "all"
        }
        default_scope = "仅当前项目"
        sel = st.selectbox(
            scope_label,
            list(scope_options.keys()),
            index=list(scope_options.keys()).index(default_scope),
            key="scope_sel_newproj"
        )
        st.session_state["scope_newproj"] = scope_options[sel]

        c1, c2 = st.columns([3, 2])
        with c1:
            title = st.text_input("项目名称")
            tags_sel = st.multiselect("项目标签(可多选)", TAG_OPTIONS)
            scene_sel = st.selectbox("场合", SCENE_OPTIONS, index=0)
        with c2:
            translation_type = st.selectbox("翻译方式", ["使用术语库", "纯机器翻译"])
            translation_mode = st.radio("模式", ["标准模式", "术语约束模式"], horizontal=True)
            prompt_text = st.text_area(
                "翻译提示(注入模型 System Prompt)",
                placeholder="写下对 DeepSeek 的硬性/优先级指令.如:时态统一为过去式.专有名词保持原文……",
                height=120,
                key="new_proj_prompt"
            )
        # === 领域自动绑定逻辑 ===
        # 若用户选择了多个标签.则默认以第一个标签为领域
        domain_val = tags_sel[0] if tags_sel else None

        # === 提交按钮 ===
        submitted = st.form_submit_button("💾 创建项目")
        if submitted:
            if not title:
                st.error("请填写项目名称")
            else:
                try:
                    # 确保 items 表存在 domain 字段
                    cur.execute("PRAGMA table_info(items);")
                    cols = [r[1] for r in cur.fetchall()]
                    if "domain" not in cols:
                        cur.execute("ALTER TABLE items ADD COLUMN domain TEXT;")
                        conn.commit()

                    # 插入新项目(含 domain)
                    cur.execute("""
                        INSERT INTO items(title, body, tags, domain,type)
                        VALUES (?, ?, ?, ?, 'project')
                    """, (
                        title,
                        prompt_text or "",
                        ",".join(tags_sel or []),
                        domain_val
                    ))
                    conn.commit()

                    st.success(f"✅ 项目 '{title}' 已创建(领域:{domain_val or '未指定'})")
                except Exception as e:
                    st.error(f"❌ 创建项目失败: {e}")

    rows = cur.execute(
        """
        SELECT
            i.id,
            i.title,
            COALESCE(i.tags,'')              AS tags,
            COALESCE(MIN(e.src_path),'')     AS src_path,
            COALESCE(i.created_at,'')        AS created_at,
            COALESCE(i.scene,'')             AS scene,
            COALESCE(i.prompt,'')            AS prompt,
            COALESCE(i.mode,'')              AS mode,
            COALESCE(i.trans_type,'')        AS trans_type
        FROM items i
        LEFT JOIN item_ext e ON e.item_id = i.id
        WHERE COALESCE(i.type,'')='project'
        GROUP BY i.id
        ORDER BY i.id DESC
        """
    ).fetchall()

    if not rows:
        st.info("暂无项目")
    else:
        # 用于收集本轮勾选的项目(ID + 文件路径)
        batch_to_delete = []

        for pid, title, tags_str, path, ct, scene, prompt_ro, mode, trans_type in rows:
            ensure_legacy_file_record(cur, conn, pid, path or None)
            file_records = fetch_project_files(cur, pid)
            tag_display = tags_str or "无"
            file_display = f"{len(file_records)} 个文件" if file_records else "无"
            selected_src_path = None

            with st.expander(f"{title}｜方式:{mode or '未设'}｜标签:{tag_display}｜场合:{scene or '未填'}｜文件:{file_display}｜创建:{ct}"):
                # ✅ 批量操作用的勾选框
                sel = st.checkbox("选择此项目(用于批量删除)", key=f"sel_proj_{pid}")
                if sel:
                    batch_to_delete.append(pid)

                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.selectbox("翻译方向", ["中译英", "英译中"], key=f"lang_{pid}")
                with c2:
                    max_len = st.number_input("分块长度", 600, 2000, 1200, 100, key=f"len_{pid}")
                with c3:
                    use_terms = st.checkbox("使用术语库", value=(mode == "术语约束模式"), key=f"ut_{pid}")

                st.caption(f"标签:{tag_display}")
                st.caption(f"场合:{scene or '未填写'}")
             
                # === 领域(domain)设置:跟随第一个标签 或 手动选择 ===
                # 读取当前项目的 domain / tags
                # 保底:items 表若没有 domain 列.动态补列(兼容旧库)
                cols_items = [r[1] for r in cur.execute("PRAGMA table_info(items)").fetchall()]
                if "domain" not in cols_items:
                    try:
                        cur.execute("ALTER TABLE items ADD COLUMN domain TEXT;")
                        conn.commit()
                    except Exception:
                        pass  # 并发或已有列时忽略
 
                row = cur.execute(
                    "SELECT IFNULL(domain,''), IFNULL(tags,'') FROM items WHERE id=?",
                    (pid,)
                ).fetchone()
                domain0, tags0 = (row or ["", ""])
                tags_list = [t for t in (tags0.split(",") if tags0 else []) if t]

                DOMAIN_OPTIONS = ["政治", "经济", "文化", "文物", "金融", "法律"]

                dom_mode = st.radio(
                    "领域设置方式",
                    ["跟随第一个标签", "手动选择"],
                    horizontal=True,
                    key=f"dom_mode_{pid}"
                )

                if dom_mode == "跟随第一个标签":
                    domain_val = (tags_list[0] if tags_list else (domain0 or None))
                    st.caption(f"当前领域(自动):{domain_val or '未指定'}(由第一个标签决定)")
                else:
                    idx = DOMAIN_OPTIONS.index(domain0) if domain0 in DOMAIN_OPTIONS else 0
                    domain_val = st.selectbox(
                        "领域(手动选择)",
                        DOMAIN_OPTIONS,
                        index=idx,
                        key=f"dom_sel_{pid}"
                    )

                sync_corpus = st.checkbox(
                    "同时回填该项目下语料的领域(仅补空或原领域相同时覆盖)",
                    value=True,
                    key=f"sync_corpus_{pid}"
                )

                if st.button("💾 保存领域设置", key=f"save_dom_{pid}", type="secondary"):
                    try:
                        # 确保 items.domain 存在
                        cols_items = [r[1] for r in cur.execute("PRAGMA table_info(items)").fetchall()]
                        if "domain" not in cols_items:
                            cur.execute("ALTER TABLE items ADD COLUMN domain TEXT;")
                            conn.commit()

                        # 更新 items.domain
                        cur.execute("UPDATE items SET domain=? WHERE id=?", (domain_val, pid))
                        conn.commit()

                        # 同步语料库的 domain(优先 corpus_main.退回 corpus)
                        def _table_exists(tb: str) -> bool:
                            return bool(cur.execute(
                                "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (tb,)
                            ).fetchone())

                        corpus_tb = "corpus_main" if _table_exists("corpus_main") else ("corpus" if _table_exists("corpus") else None)
                        if sync_corpus and corpus_tb and domain_val:
                            # 确保列存在
                            cols_corpus = [r[1] for r in cur.execute(f"PRAGMA table_info({corpus_tb})").fetchall()]
                            if "domain" not in cols_corpus:
                                cur.execute(f"ALTER TABLE {corpus_tb} ADD COLUMN domain TEXT;")
                                conn.commit()

                            # 仅补空或原 domain 与 domain0 相同时覆盖.避免误伤跨领域数据
                            cur.execute(f"""
                                UPDATE {corpus_tb}
                                SET domain = ?
                                WHERE project_id = ?
                                AND (domain IS NULL OR TRIM(domain)='' OR domain = ?)
                            """, (domain_val, pid, domain0 or ""))
                            conn.commit()

                        st.success("✅ 已保存领域设置")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 保存失败:{e}")

                if prompt_ro:
                    try:
                        cur.execute("SELECT IFNULL(prompt, '') FROM items WHERE id=?", (pid,))
                        prompt_ro = (cur.fetchone() or [""])[0]
                    except Exception:
                        prompt_ro = ""

                    st.text_area("翻译提示(只读)", prompt_ro or "", height=120, key=f"proj_prompt_ro_{pid}")

                file_col, action_col = st.columns([3, 1])

                with file_col:
                    if file_records:
                        option_labels = []
                        option_map = {}
                        for rec in file_records:
                            label = f"[#{rec['id']}] {rec['name']}"
                            if rec["uploaded_at"]:
                                label += f"｜{rec['uploaded_at']}"
                            option_labels.append(label)
                            option_map[label] = rec
                        sel_key = f"file_sel_{pid}"
                        default_label = st.session_state.get(sel_key)
                        if default_label not in option_labels:
                            default_label = option_labels[0]
                            st.session_state[sel_key] = default_label
                        chosen_label = st.selectbox(
                            "选择要翻译的源文件",
                            option_labels,
                            index=option_labels.index(default_label),
                            key=sel_key,
                        )
                        selected_src_path = option_map[chosen_label]["path"]
                        st.caption(f"已上传 {len(file_records)} 个附件，当前选中:{option_map[chosen_label]['name']}")
                    else:
                        selected_src_path = path or None
                        st.info("尚未上传源文件，可在下方上传一个或多个文件。")                      

                    upload_key = f"up_multi_{pid}"
                    processed_key = f"up_multi_processed_{pid}"
                    if upload_key not in st.session_state:
                        st.session_state[processed_key] = set()
                    uploads = st.file_uploader(
                        "新增/补传文件(可多选)",
                        type=["txt", "docx", "xlsx", "pdf"],
                        accept_multiple_files=True,
                        key=upload_key
                    )
                    if uploads:
                        processed_names = st.session_state.setdefault(processed_key, set())
                        saved = 0
                        for uf in uploads:
                            if not uf or uf.name in processed_names:
                                continue
                            data = uf.read()
                            new_path = register_project_file(cur, conn, pid, uf.name, data)
                            if new_path:
                                cur.execute("SELECT id FROM item_ext WHERE item_id=?", (pid,))
                                r = cur.fetchone()
                                if r:
                                    cur.execute("UPDATE item_ext SET src_path=? WHERE id=?", (new_path, r[0]))
                                else:
                                    cur.execute("INSERT INTO item_ext (item_id, src_path) VALUES (?, ?)", (pid, new_path))
                                conn.commit()
                                saved += 1
                                processed_names.add(uf.name)
                        if saved:
                            st.success(f"✅ 已上传 {saved} 个文件")
                    else:
                        st.session_state.pop(processed_key, None)

                    if file_records:
                        st.markdown("附件列表:")
                        for rec in file_records:
                            info_cols = st.columns([5, 1])
                            info = f"[#{rec['id']}] {rec['name']}｜{os.path.basename(rec['path'])}"
                            if rec["uploaded_at"]:
                                info += f"｜{rec['uploaded_at']}"
                            info_cols[0].write(info)
                            if info_cols[1].button("删除", key=f"del_file_{rec['id']}"):
                                remove_project_file(cur, conn, rec["id"])
                                st.rerun()

                with action_col:
                    if st.button("删除项目", key=f"del_proj_{pid}"):
                        cleanup_project_files(cur, conn, pid)
                        cur.execute("DELETE FROM items WHERE id=?", (pid,))
                        cur.execute("DELETE FROM item_ext WHERE item_id=?", (pid,))
                        conn.commit()
                        st.success("项目已删除")
                        st.rerun()

                # —— 执行翻译
                if st.button("执行翻译", key=f"run_{pid}", type="primary"):
                    run_project_translation_ui(
                        pid=pid,
                        project_title=title,
                        src_path=selected_src_path,
                        conn=conn,
                        cur=cur
                    )

                # —— 新增：进入翻译工作台（可编辑）
                if st.button("进入翻译工作台（可编辑）", key=f"workspace_{pid}", type="secondary"):
                    # 1) 环境检查：有源文件吗？
                    if not selected_src_path or not os.path.exists(selected_src_path):
                        st.error("缺少源文件，请先在上面选择或上传源文件。")
                        st.stop()
                    st.session_state[f"workspace_activated_{pid}"] = True

                    # 2) 读取源文本（这里定义 src_text）
                    src_text = read_source_file(selected_src_path)

                    # 3) 分段
                    blocks = split_paragraphs(src_text)
                    if not blocks:
                        st.error("源文件内容为空，或未识别到有效段落")
                        st.stop()

                    # 4) 术语：用统一接口 + 转成 term_pairs 供高亮用
                    term_map_all, term_meta = get_terms_for_project(cur, pid, use_dynamic=True)
                    term_pairs = list(term_map_all.items())

                    # 5) 用统一管线 translate_block_with_kb 做初译
                    ak, model = get_deepseek()
                    if not ak:
                        st.error("未检测到 DeepSeek Key，请配置 deepseek")
                        st.stop()

                    # 翻译方向（沿用你项目里已有的变量）
                    lang_pair_val = st.session_state.get(f"lang_{pid}", "中译英")

                    # 是否启用语义召回 / 召回范围，直接沿用上面 Tab1 的设置
                    use_semantic_val = use_semantic
                    scope_val_local = scope_label

                    draft = []
                    for blk in blocks:
                        blk = (blk or "").strip()
                        if not blk:
                            draft.append("")
                            continue

                        res = translate_block_with_kb(
                            cur=cur,
                            project_id=pid,
                            block_text=blk,
                            lang_pair=lang_pair_val,
                            ak=ak,
                            model=model,
                            use_semantic=use_semantic_val,
                            scope=scope_val_local,
                            fewshot_examples=None,  # 工作台模式先不注入 few-shot
                        )
                        draft.append(res["tgt"])

                    # 6) 保存到 session_state，供下面编辑界面使用
                    st.session_state[f"workspace_src_{pid}"] = blocks
                    st.session_state[f"workspace_draft_{pid}"] = draft
                    st.session_state[f"workspace_terms_{pid}"] = term_pairs

                    st.success("草稿已生成，请下方开始编辑 ↓")


                # ③ 翻译工作台 UI：只有当 session 里有草稿时才显示
                if st.session_state.get(f"workspace_draft_{pid}") and st.session_state.get(f"workspace_activated_{pid}", False):

                    st.markdown("## 📝 翻译工作台（可编辑）")

                    # 从 session 中取回草稿和术语
                    blocks = st.session_state.get(f"workspace_src_{pid}", [])
                    draft  = st.session_state.get(f"workspace_draft_{pid}", [])
                    terms  = st.session_state.get(f"workspace_terms_{pid}", [])

                    if not blocks or not draft:
                        st.info("当前暂无草稿，请先点击“进入翻译工作台（可编辑）”生成初稿。")
                    else:
                        edited_blocks = []

                        for i, (src, trg) in enumerate(zip(blocks, draft), 1):
                            st.markdown(f"### 段落 {i}")

                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**原文**")
                                st.markdown(
                                    f"<div style='padding:8px;border:1px solid #ccc;background:#f8f8f8'>{src}</div>",
                                    unsafe_allow_html=True
                                )

                            with col2:
                                st.markdown("**译文（可编辑）**")
                                new_trg = st.text_area(
                                    label="编辑后的译文",
                                    value=trg,
                                    key=f"edit_{pid}_{i}",
                                    height=120
                                )
                                edited_blocks.append(new_trg)

                                # 术语高亮（如果你前面已经定义了 highlight_terms）
                                if "highlight_terms" in globals():
                                    highlighted = highlight_terms(new_trg, terms)
                                    st.markdown("术语高亮：")
                                    st.markdown(
                                        f"<div style='padding:8px;border:1px solid #ccc;background:#f0fff0'>{highlighted}</div>",
                                        unsafe_allow_html=True
                                    )

                        # —— 确认生成最终译文 —— 
                        if st.button("✅ 确认生成最终译文", key=f"confirm_{pid}", type="primary"):
                            final_text = "\n\n".join(edited_blocks)

                            # 语言方向从 session 里拿（跟你翻译时保持一致）
                            lang_pair_val = st.session_state.get(f"lang_{pid}", "中译英")

                            cur.execute("""
                                INSERT INTO trans_ext (project_id, src_path, lang_pair, mode, output_text, created_at)
                                VALUES (?, ?, ?, ?, ?, datetime('now'))
                            """, (pid, selected_src_path, lang_pair_val, "工作台模式", final_text))
                            conn.commit()

                            st.success("最终译文已生成并写入历史！")

                            # 清空工作台草稿
                            st.session_state.pop(f"workspace_src_{pid}", None)
                            st.session_state.pop(f"workspace_draft_{pid}", None)
                            st.session_state.pop(f"workspace_terms_{pid}", None)
 
        # —— 批量删除按钮(在项目列表底部)
        if batch_to_delete:
            st.warning(f"已勾选 {len(batch_to_delete)} 个项目，操作不可撤销。")
            if st.button("🗑️ 批量删除选中项目", key="batch_del_projects"):
                deleted = 0
                for pid_del in batch_to_delete:
                    cleanup_project_files(cur, conn, pid_del)
                    cur.execute("DELETE FROM items WHERE id=?", (pid_del,))
                    cur.execute("DELETE FROM item_ext WHERE item_id=?", (pid_del,))
                    deleted += 1
                conn.commit()
                st.success(f"已批量删除 {deleted} 个项目")
                st.rerun()
        else:
            st.caption("提示:如需批量删除，可在上方勾选多个项目。")

# ========== Tab2:术语库管理 ==========
elif choice.startswith("📘"):
    render_term_management(st, cur, conn, BASE_DIR, key_prefix="term")


# ========== Tab3:翻译历史(增强版) ==========
elif choice.startswith("📊"):
    st.subheader("📊 翻译历史记录(可写入语料 / 抽取术语 / 下载对照 / 删除)")

    rows = cur.execute("""
        SELECT id, project_id, lang_pair,
               substr(IFNULL(output_text,''),1,120) AS prev, created_at
        FROM trans_ext
        ORDER BY datetime(created_at) DESC
        LIMIT 200
    """).fetchall()

    if not rows:
        st.info("暂无历史记录。")
    else:
        for rid, pid, lp, prev, ts in rows:
            # 项目标题(做语料标题/展示)
            ttl_row = cur.execute("SELECT IFNULL(title,'') FROM items WHERE id=?", (pid,)).fetchone()
            proj_title = (ttl_row or [""])[0] or f"project#{pid}"
            proj_domain = _project_domain(pid, cur)

            with st.expander(f"#{rid}｜项目 {pid}｜{proj_title}｜{lp}｜{ts}", expanded=False):
                # 译文全文 & 源文件路径
                det = cur.execute("SELECT output_text, src_path FROM trans_ext WHERE id=?", (rid,)).fetchone()
                tgt_full, src_path = det or ("", "")
                st.code(prev or "", language="text")
                st.text_area("译文全文", tgt_full or "", height=220, key=f"hist_full_{rid}")

                # 尝试读取原文(如果当时保存了源文件路径)
                try:
                    src_full = read_source_file(src_path) if src_path else ""
                except Exception:
                    src_full = ""

                with st.expander("原文预览(若上传了源文件)", expanded=False):
                    st.text_area("原文全文", src_full or "(未保存/未上传源文件)", height=160, key=f"hist_src_{rid}")

                c1, c2, c3, c4, c5 = st.columns(5)

                # 1) 添加进语料库 / 添加+重建索引
                with c1:
                    # 1-1 只写入语料库
                    if st.button("➕ 添加进语料库", key=f"hist_add_corpus_{rid}"):
                        if not src_full and not tgt_full:
                            st.warning("原文和译文都为空，无法写入语料库。")
                        else:
                            cur.execute("""
                                INSERT INTO corpus (title, project_id, lang_pair, src_text, tgt_text, note, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                            """, (
                                f"{proj_title} · history#{rid}",
                                pid,
                                lp or "",
                                src_full or None,
                                tgt_full or "",
                                f"from trans_ext#{rid}",
                            ))
                            conn.commit()
                            st.success("✅ 已写入语料库")

                    # 1-2 写入语料库并重建索引
                    if st.button("➕ 添加并重建索引", key=f"hist_add_corpus_rebuild_{rid}_{idx}"):
                        if not src_full and not tgt_full:
                            st.warning("原文和译文都为空，无法写入语料库。")
                        else:
                            # 先写入语料库
                            cur.execute("""
                                INSERT INTO corpus (title, project_id, lang_pair, src_text, tgt_text, note, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                            """, (
                                f"{proj_title} · history#{rid}",
                                pid,
                                lp or "",
                                src_full or None,
                                tgt_full or "",
                                f"from trans_ext#{rid}",
                            ))
                            conn.commit()

                            # 再重建该项目的语义索引
                            res_idx = rebuild_project_semantic_index(cur, pid, split_fn=_split_pair_for_index)
                            if res_idx.get("ok"):
                                st.success(
                                    f"✅ 已写入语料库并重建索引: 新增 {res_idx['added']} 条, 总量 {res_idx['total']} 条"
                                )
                            else:
                                st.warning(
                                    f"已写入语料库，但重建索引失败: {res_idx.get('msg','未知错误')}"
                                )

                # 2) 提取术语(优先语料库同款模型，缺省回退 DeepSeek)
                with c2:
                    if st.button("🧠 提取术语", key=f"hist_extract_terms_{rid}"):
                        ak, model = get_deepseek()
                        if not ak:
                            st.info("未检测到 DeepSeek Key，将仅使用语料库同款模型进行抽取。")

                        # 合并原文+译文.提高候选质量
                        big = ((src_full or "") + "\n" + (tgt_full or "")).strip()
                        res = ds_extract_terms(
                            big,
                            ak,
                            model,
                            src_lang="zh",
                            tgt_lang="en",
                            prefer_corpus_model=True,
                            default_domain=proj_domain,
                        )
                        res, dup_terms = dedup_terms_against_db(cur, res, pid)
                        if not res:
                            st.info("未抽取到术语或解析失败")
                        else:
                            ins_term = ins_corpus = 0
                            for o in res:
                                domain_val = (o.get("domain") or proj_domain or "其他")
                                strategy_val = (o.get("strategy") or "history-extract")
                                cur.execute("""
                                    INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """, (
                                    o.get("source_term") or "",
                                    (o.get("target_term") or None),
                                    domain_val,
                                    pid,
                                    strategy_val,
                                    (o.get("example") or None),
                                ))
                                ins_term += 1

                                src_ex, tgt_ex = _locate_example_pair(o.get("example"), src_full, tgt_full)
                                if src_ex:
                                    cur.execute(
                                        """
                                        INSERT INTO corpus(title, project_id, lang_pair, src_text, tgt_text, note, domain, source, created_at)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                                        """,
                                        (
                                            f"{proj_title} · term#{rid}",
                                            pid,
                                            lp or "",
                                            src_ex,
                                            tgt_ex,
                                            "term-example",
                                            domain_val,
                                            "history-term",
                                        ),
                                    )
                                    ins_corpus += 1
                            conn.commit()
                            msg = f"✅ 已写入术语库 {ins_term} 条，同步语料库 {ins_corpus} 条"
                            if dup_terms:
                                msg += f"；跳过重复 {len(dup_terms)} 条"
                            st.success(msg)

                # 3) 下载双语对照(CSV / DOCX)
                with c3:
                    if st.button("⬇️ CSV 对照", key=f"hist_dl_bicsv_btn_{rid}"):
                        if not src_full:
                            st.warning("找不到原文(未上传源文件).无法生成 CSV 对照")
                        else:
                            try:
                                csv_name = f"bilingual_history_{rid}.csv"
                                csv_bytes = export_csv_bilingual((src_full, tgt_full),
                                    filename=f"bilingual_history_{rid}.csv"
                                )
                            except TypeError:
                                # 如果你的导出函数是 text→bytes 版本
                                csv_name = f"bilingual_history_{rid}.csv"
                                csv_bytes = export_csv_bilingual(src_full, tgt_full)
                            st.download_button("下载 CSV", data=csv_bytes,
                                               file_name=csv_name, mime="text/csv",
                                               key=f"hist_dl_bicsv_{rid}")

                with c4:
                    if st.button("⬇️ DOCX 对照", key=f"hist_dl_bidocx_btn_{rid}"):
                        if not src_full:
                            st.warning("找不到原文(未上传源文件).无法生成 DOCX 对照")
                        else:
                            try:
                                docx_path = export_docx_bilingual(
                                    filename=f"bilingual_history_{rid}.docx"
                                )
                                with open(docx_path, "rb") as f:
                                    data_docx = f.read()
                            except TypeError:
                                # 如果你的导出函数是 text→bytes 版本
                                data_docx = export_docx_bilingual(src_full, tgt_full)
                            st.download_button("下载 DOCX", data=data_docx,
                                               file_name=f"bilingual_history_{rid}.docx",
                                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                               key=f"hist_dl_bidocx_{rid}")

                # 4) 🗑 删除本条历史(安全确认)
                with c5:
                    with st.expander("🗑 删除本条历史(不可恢复)", expanded=False):
                        st.warning("此操作将永久删除该条 trans_ext 记录(不影响已写入语料库/术语表的数据)。")
                        ok = st.checkbox(f"我确认删除 #{rid}", key=f"hist_del_ck_{rid}")
                        if st.button("确认删除", key=f"hist_del_btn_{rid}") and ok:
                            cur.execute("DELETE FROM trans_ext WHERE id=?", (rid,))
                            conn.commit()
                            st.success("已删除.请刷新页面查看结果。")
                            st.stop()  # 终止本次渲染.避免在已删除数据上继续操作

                # 原有的“下载译文 TXT”
                st.download_button("下载译文 (TXT)", tgt_full or "",
                                   file_name=f"history_{rid}.txt",
                                   mime="text/plain",
                                   key=f"hist_dl_txt_{rid}")

# ========== Tab4:语料库管理 ==========
elif choice.startswith("📚"):
    def render_corpus_manager(st, cur, conn, pid_prefix="corpus"):
        st.header("📚 语料库管理")
        sk = make_sk(pid_prefix)

    render_corpus_manager(st, cur, conn)

    # 初始化 Few-shot 状态字典
    get_project_ref_ids(None)
    get_project_fewshot_enabled(None)
    st.session_state.setdefault("corpus_target_project", None)
    st.session_state.setdefault("corpus_target_label", "(请选择 Few-shot 目标项目)")

    sub = st.tabs(["新建语料", "浏览/检索", "使用与导出"])

    # -------- 新建语料 --------
    with sub[0]:
        st.subheader("📥 上传 / 对齐 / 入库")

        colA, colB = st.columns(2)
        with colA:
            one_file = st.file_uploader("① 单个文件(DOCX 表格对照 / 单语 DOCX/TXT/PDF)",
                                        type=["docx", "txt", "pdf"], key="up_one")
        with colB:
            two_src = st.file_uploader("② 原文文件(可选:与 ③ 搭配做对齐)",
                                       type=["docx", "txt", "csv", "pdf"], key="up_src")
            two_tgt = st.file_uploader("③ 译文文件(可选:与 ② 搭配做对齐)",
                                       type=["docx", "txt", "csv", "pdf"], key="up_tgt")

        st.divider()
        meta1, meta2, meta3 = st.columns([2, 1, 1])
        with meta1:
            title = st.text_input("语料标题", value="未命名语料")
        with meta2:
            lp = st.selectbox("方向", ["自动", "中译英", "英译中"])
        with meta3:
            pid_val = st.text_input("项目ID(可留空)")
        pid = int(pid_val) if pid_val.strip().isdigit() else None

        ins = 0
        pairs = []
        src_text = tgt_text = ""
        preview_df = None

        # ========== 路径 A:单个文件 ==========
        if one_file is not None and (two_src is None and two_tgt is None):
            ext = (one_file.name.split(".")[-1] or "").lower()
            bio = io.BytesIO(one_file.getvalue())

            if ext == "docx":
                # 先尝试“表格对照”
                tables = read_docx_tables_info(io.BytesIO(bio.getvalue()))
                if tables:
                    st.caption("检测到 DOCX 表格，优先作为双语对照导入。")
                    # 简单起见，默认第 0 张表的第 0/1 列;你也可以加入下拉选择
                    pairs = extract_pairs_from_docx_table(
                        io.BytesIO(bio.getvalue()),
                        table_index=0,
                        src_col=0,
                        tgt_col=1
                    )
                else:
                    # 没有表格 → 单语文本
                    src_text = read_docx_text(io.BytesIO(bio.getvalue()))

            elif ext == "txt":
                src_text = read_txt(bio)

            elif ext == "pdf":
                src_text = read_pdf_text(io.BytesIO(bio.getvalue()))

        # ========== 路径 B:两个文件(原文 + 译文)==========
        elif two_src is not None and two_tgt is not None:
            def read_any(f):
                e = (f.name.split(".")[-1] or "").lower()
                b = io.BytesIO(f.getvalue())
                if e == "docx":
                    return read_docx_text(b)
                if e == "txt":
                    return read_txt(b)
                if e == "csv":
                    try:
                        df = pd.read_csv(b)
                        return "\n".join(df.iloc[:, 0].astype(str).fillna(""))
                    except Exception:
                        return ""
                if e == "pdf":
                    return read_pdf_text(b)
                return ""
            src_text = read_any(two_src)
            tgt_text = read_any(two_tgt)

        # ========== 预览与决定入库方式 ==========
        # 情况 1:有 pairs(来自 DOCX 表格)
        if pairs:
            st.success(f"解析到 {len(pairs)} 对(DOCX 表格)")
            preview_df = pd.DataFrame(pairs[:200], columns=["源句", "目标句"])

        # 情况 2:没有 pairs，但拿到了 src/tgt 文本 → 切句/对齐
        elif src_text and tgt_text:
            sents_src = split_sents(src_text, "zh" if lp.startswith("中") else "auto")
            sents_tgt = split_sents(tgt_text, "en" if lp.startswith("英") else "auto")
            st.caption(f"将对齐: src={len(sents_src)}  tgt={len(sents_tgt)}")
            if st.button("🔎 执行语义对齐", key="do_align"):
                pairs_aligned = align_semantic(sents_src, sents_tgt, max_jump=5)
                st.info(f"对齐得到 {len(pairs_aligned)} 对")
                pairs = [(s, t) for (s, t, score) in pairs_aligned]
                if pairs:
                    preview_df = pd.DataFrame(pairs[:200], columns=["源句", "目标句"])

        # 情况 3:只有单语文本(PDF/DOCX/TXT)
        elif src_text and not tgt_text:
            sents_src = split_sents(src_text, "zh" if lp.startswith("中") else "auto")
            st.info(f"检测到单语文本，共 {len(sents_src)} 句。将以单语语料写入(译文为空)。")
            preview_df = pd.DataFrame(
                [{"源句": s, "目标句": ""} for s in sents_src[:200]]
            )

        if preview_df is not None:
            st.dataframe(preview_df, width='stretch')

        # —— 按钮 + 选项:导入语料库 | 同时重建索引
        c_imp, c_opt, c_build = st.columns([1, 1, 1])
        do_import = c_imp.button("📥 写入语料库", type="primary", key="write_pairs_btn")
        do_build_opt = c_opt.checkbox("导入后立即重建索引", value=True, key="build_vec_opt")
        only_build_now = c_build.button("🧠 仅重建索引(不导入)", key="only_build")

        st.caption("提示: 索引也可以稍后在“使用与索引 / 导出”页的 C 区统一重建。")

        # ===== 这里开始用统一管线 =====
        # 统一算一个兜底标题（优先用界面上的 title，再用文件名）
        default_title = ""
        if one_file is not None:
            default_title = one_file.name
        elif two_src is not None:
            default_title = two_src.name

        if do_import:
            import_corpus_from_upload(
                st,
                cur,
                conn,
                pid=pid,              # 你上面 meta 区的项目 ID
                title=title,          # 你上面输入的语料标题
                lp=lp,                # 方向选择
                pairs=pairs,
                src_text=src_text,
                tgt_text=tgt_text,
                default_title=default_title,
                build_after_import=do_build_opt,
            )

        if only_build_now:
            # 仅重建当前项目的语义索引(不导入新语料)
            if pid:
                res_idx = rebuild_project_semantic_index(cur, pid, split_fn=_split_pair_for_index)
                if res_idx.get("ok"):
                    st.success(
                        f"🧠 索引已重建: 新增 {res_idx['added']}，总量 {res_idx['total']}。"
                    )
                else:
                    st.error(f"重建失败: {res_idx.get('msg','未知错误')}")
            else:
                st.warning("请先在上方填写有效的项目ID，再重建索引。")

    # -------- 浏览/检索 --------
    with sub[1]:
        st.subheader("🔎 浏览/检索")
        k1, k2, k3 = st.columns([2, 1, 1])
        with k1:
            kw = st.text_input("关键词(标题/备注/译文)", "", key=sk("kw"))
        with k2:
            lp_filter = st.selectbox("方向过滤", ["全部", "中译英", "英译中", "自动"], key=sk("lp_filter"))
        with k3:
            limit = st.number_input("条数", min_value=10, max_value=1000, value=200, step=10, key=sk("limit"))

        sql = """
        SELECT id, title, IFNULL(project_id,''), IFNULL(lang_pair,''), 
               substr(IFNULL(tgt_text,''),1,80), created_at
        FROM corpus
        WHERE 1=1
        """
        params = []
        if kw.strip():
            like = f"%{kw.strip()}%"
            sql += " AND (title LIKE ? OR IFNULL(note,'') LIKE ? OR IFNULL(tgt_text,'') LIKE ?)"
            params.extend([like, like, like])
        if lp_filter != "全部":
            sql += " AND lang_pair=?"
            params.append(lp_filter)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        rows = cur.execute(sql, params).fetchall()
        if not rows:
            st.info("暂无匹配语料。")

        else:
            for rid, ttl, pj, lpv, prev, ctime in rows:
                with st.expander(f"[{rid}] {ttl} | 项目:{pj} | 方向:{lpv} | {ctime}"):
                    st.write(f"**ID**: {rid}  **项目ID**: {pj}  **方向**: {lpv}  **时间**: {ctime}")
                    st.write(f"**预览(译文前80字)**: {prev}")
                    det = cur.execute(
                        "SELECT IFNULL(src_text,''), IFNULL(tgt_text,'') FROM corpus WHERE id=?",
                        (rid,)
                    ).fetchone()
                    src_all, tgt_all = det or ("", "")
                    st.text_area("源文", src_all, height=160, key=sk(f"src_{rid}"))
                    st.text_area("译文", tgt_all, height=160, key=sk(f"tgt_{rid}"))

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        if st.button("加入参考集合", key=sk(f"add_ref_{rid}")):
                            target_pid = st.session_state.get("corpus_target_project")
                            if not target_pid:
                                st.warning("请先在上方选择 Few-shot 目标项目，再添加参考语料。")
                            else:
                                refs = get_project_ref_ids(target_pid)
                                refs.add(int(rid))
                                st.success(f"✅ 已加入项目 #{target_pid} 的参考集合(“使用与导出”查看)")
                    with c2:
                        if st.button("导出TXT", key=sk(f"cor_txt_{rid}")):
                            st.download_button(
                                "下载译文TXT",
                                tgt_all or "",
                                file_name=f"corpus_{rid}.txt",
                                mime="text/plain",
                                key=sk(f"cor_txt_dl_{rid}")
                            )
                    with c3:
                        if st.button("导出CSV(中英对照)", key=sk(f"cor_csv_{rid}")):
                            df_out = pd.DataFrame(
                                [{"source": src_all, "target": tgt_all}]
                            )
                            csv_data = df_out.to_csv(index=False)
                            st.download_button(
                                "下载CSV",
                                csv_data,
                                file_name=f"corpus_{rid}.csv",
                                mime="text/csv",
                                key=sk(f"cor_csv_dl_{rid}")
                            )
                    with c4:
                        if st.button("删除", key=sk(f"del_{rid}")):
                            cur.execute("DELETE FROM corpus WHERE id=?", (rid,))
                            conn.commit()
                            st.warning("🗑️ 已删除，刷新后生效")
                            st.rerun()

    # -------- 使用与导出 --------
    with sub[2]:
        st.subheader("🧩 使用与索引 / 导出")

        proj_rows = cur.execute("SELECT id, IFNULL(title,'(未命名)') FROM items WHERE COALESCE(type,'')='project' ORDER BY id DESC LIMIT 200").fetchall()
        proj_options = {"(请选择 Few-shot 目标项目)": None}
        proj_options.update({f"[{pid}] {ttl}": pid for pid, ttl in proj_rows})
        option_labels = list(proj_options.keys())
        saved_label = st.session_state.get("corpus_target_label")
        if saved_label not in option_labels:
            saved_label = option_labels[0]
        selection_label = st.selectbox(
            "Few-shot 参考集合将绑定到哪个项目？",
            option_labels,
            index=option_labels.index(saved_label),
            key="corpus_proj_select",
        )
        st.session_state["corpus_target_label"] = selection_label
        st.session_state["corpus_target_project"] = proj_options.get(selection_label)
        if st.session_state["corpus_target_project"]:
            st.caption(f"当前 Few-shot 目标: {selection_label}")
        else:
            st.caption("未选择项目:请先在此指定目标项目，再去其他子页添加参考示例。")

        # A 区:参考集合合并与导出
        target_pid = st.session_state.get("corpus_target_project")
        if not target_pid:
            st.info("请先通过页面上方的下拉框选择 Few-shot 目标项目，再管理参考集合。")
        else:
            ids = sorted({int(x) for x in get_project_ref_ids(target_pid)}, reverse=True)
            st.caption(f"项目 #{target_pid} 的已选参考数: {len(ids)}")
            if ids:
                qmarks = ",".join(["?"] * len(ids))
                dets = cur.execute(
                    f"SELECT id, title, lang_pair, IFNULL(src_text,''), IFNULL(tgt_text,'') "
                    f"FROM corpus WHERE id IN ({qmarks})",
                    ids
                ).fetchall()
                order_map = {rid: idx for idx, rid in enumerate(ids)}
                dets.sort(key=lambda row: order_map.get(row[0], len(order_map)))
                merged_demo = "\n\n---\n\n".join(
                    [f"\n\n源文:\n{src}\n\n译文:\n{tgt}" for (_, _, _, src, tgt) in dets]
                )
                st.text_area("合并预览", merged_demo, height=240, key=sk("merge_preview"))

                cxa, cxb = st.columns(2)
                with cxa:
                    if st.button("清空参考集合", key=sk(f"clear_refs_{target_pid}")):
                        refs = get_project_ref_ids(target_pid)
                        refs.clear()
                        st.info(f"已清空项目 #{target_pid} 的参考集合")
                with cxb:
                    if st.button("导出参考TXT", key=sk(f"export_refs_txt_{target_pid}")):
                        st.download_button(
                            "下载TXT",
                            merged_demo,
                            file_name=f"corpus_refs_{target_pid}.txt",
                            mime="text/plain",
                            key=sk(f"export_refs_txt_dl_{target_pid}")
                        )
            else:
                st.info("还没有选择任何参考语料，可以在“浏览/检索”中勾选后再来。")

        st.markdown("---")

        # B 区:Few-shot 注入开关
        if not target_pid:
            st.info("选择目标项目后才能配置 Few-shot 注入开关。")
            use_fs = False
        else:
            curr_state = get_project_fewshot_enabled(target_pid)
            use_fs = st.checkbox(
                "翻译时自动注入这些参考语料作为 Few-shot 提示",
                value=curr_state,
                key=sk(f"use_fs_{target_pid}")
            )
        if st.button("保存参考注入开关", key=sk("save_fs")):
            if not target_pid:
                st.warning("请先选择 Few-shot 目标项目")
            else:
                set_project_fewshot_enabled(target_pid, use_fs)
                st.success(
                    f"已更新:项目 #{target_pid} "
                    + ("将注入参考 few-shot" if use_fs else "不会自动注入参考")
                )

        st.markdown("---")

        # C 区:项目级索引管理 + 语义召回测试
        st.subheader("🧠 语义索引管理 & 召回测试")

        # 选择项目
        proj_rows = cur.execute(
            """
            SELECT id, title
            FROM items
            WHERE COALESCE(type,'')='project'
            ORDER BY id DESC
            LIMIT 200
            """
        ).fetchall()
        proj_map = {"(请选择)": None}
        proj_map.update({f"[{i}] {t}": i for (i, t) in proj_rows})
        proj_sel = st.selectbox("选择要测试/重建索引的项目", list(proj_map.keys()), key=sk("vec_proj"))
        pid_sel = proj_map.get(proj_sel)
        # 显示当前项目索引中 corpus / history 统计
        idx_total = idx_corpus = idx_hist = idx_other = 0
        if pid_sel:
            try:
                mode, index_obj, mapping, vecs = _load_index(int(pid_sel))
                if isinstance(mapping, list):
                    idx_total = len(mapping)
                    for m in mapping:
                        src_tag = (m.get("source") or "").lower()
                        if src_tag == "history":
                            idx_hist += 1
                        elif src_tag in ("", "corpus"):
                            idx_corpus += 1
                        else:
                            idx_other += 1
            except Exception:
                # 索引文件损坏/不存在时直接忽略统计
                pass

        if pid_sel:
            st.caption(
                f"当前索引条数: {idx_total} 条 "
                f"(语料库: {idx_corpus} 条, 来自翻译历史: {idx_hist} 条, 其他: {idx_other} 条)。"
            )
        else:
            st.caption("请选择项目以查看该项目的索引状态。")


        c_build1, c_build2 = st.columns(2)
        with c_build1:
            if st.button("构建/更新该项目索引", key=sk("build_idx_btn")):
                if pid_sel:
                    res_idx = rebuild_project_semantic_index(cur, pid_sel, split_fn=_split_pair_for_index)
                    if res_idx.get("ok"):
                        st.success(
                            f"索引已更新: 新增 {res_idx['added']}，总量 {res_idx['total']}。"
                        )
                    else:
                        st.error(f"构建失败: {res_idx.get('msg','未知错误')}")
                else:
                    st.warning("请先选择具体项目。")

        with c_build2:
            st.caption("说明:索引用于语义召回参考例句，翻译时由系统自动调用。")

        q_demo = st.text_area("试搜一句话(将以语义相似检索参考)", "", height=80, key=sk("q_demo"))
        topk = st.number_input("Top-K", 1, 10, 5, key=sk("q_topk"))
        if st.button("🔍 语义召回测试", key=sk("q_vec")):
            if pid_sel and q_demo.strip():
                hits = semantic_retrieve(pid_sel, q_demo.strip(), topk=int(topk), cur=cur)
                if not hits:
                    st.info("索引为空或未命中。请先构建索引或增加语料。")
                else:
                    for row in hits:
                        sc, m, txt = row[:3]   # 只拿前 3 个，后面多余的忽略
                        st.write(f"**{sc:.3f}** | {m.get('title','')} | {m.get('lang_pair','')}")
                        st.code(txt, language="text")
                        st.markdown("---")
            else:
                st.warning("请先选择项目并输入要检索的内容。")

elif choice.startswith("🧠"):
    render_index_manager_by_domain(st, conn, cur)

