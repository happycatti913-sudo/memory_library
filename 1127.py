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

from app_core.config import BASE_DIR, KBEmbedder, dedup_terms_against_db, make_sk, _project_domain
from app_core.database import init_db
from app_core.semantic_index import _load_index, rebuild_project_semantic_index
from app_core.semantic_ops import align_semantic, semantic_retrieve
from app_core.translation_ops import get_deepseek
from app_core.term_extraction import _locate_example_pair, ds_extract_terms
from app_core.file_ops import export_csv_bilingual, export_docx_bilingual, read_source_file
from app_core.corpus_ops import import_corpus_from_upload
from app_core.ui_common import render_table
from app_core.ui_index import render_index_manager, render_index_manager_by_domain
from app_core.ui_projects import render_project_tab
from app_core.ui_terms import render_term_management
from app_core.text_utils import _split_pair_for_index, read_docx_tables_info, read_docx_text, read_pdf_text, read_txt, split_sents

# ========== 页面设置 ==========
st.set_page_config(page_title="个人翻译知识库管理系统3.0", layout="wide")

# ========== 工具函数 ==========
# ======= 获取某条历史记录对应的原文(优先 items.body.兜底 src_path 仅作为标题提示)=======
# ======= 对齐并导出(依赖你已有的 split_blocks / align_export)=======
# ========== 路径/DB ==========
conn, cur = init_db()

# ====== 术语管理 UI ======

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

