# -*- coding: utf-8 -*-
"""翻译历史页 UI 与写入/抽取操作。"""
from __future__ import annotations

import streamlit as st

from .config import _project_domain, dedup_terms_against_db
from .database import ensure_col
from .file_ops import export_csv_bilingual, export_docx_bilingual, read_source_file
from .semantic_index import rebuild_project_semantic_index
from .term_extraction import _locate_example_pair, ds_extract_terms
from .text_utils import _split_pair_for_index
from .translation_ops import get_deepseek


def render_history_tab(st, cur, conn):
    st.subheader("📊 翻译历史记录(可写入语料 / 抽取术语 / 下载对照 / 删除)")

    try:
        ensure_col(conn, cur, "trans_ext", "lang_pair", "TEXT")
        ensure_col(conn, cur, "trans_ext", "output_text", "TEXT")
        ensure_col(conn, cur, "trans_ext", "src_path", "TEXT")
    except Exception:
        pass

    rows = cur.execute(
        """
        SELECT id, project_id, lang_pair,
               substr(IFNULL(output_text,''),1,120) AS prev, created_at
        FROM trans_ext
        ORDER BY datetime(created_at) DESC
        LIMIT 200
    """
    ).fetchall()

    if not rows:
        st.info("暂无历史记录。")
        return

    for rid, pid, lp, prev, ts in rows:
        ttl_row = cur.execute("SELECT IFNULL(title,'') FROM items WHERE id=?", (pid,)).fetchone()
        proj_title = (ttl_row or [""])[0] or f"project#{pid}"
        proj_domain = _project_domain(pid, cur)

        with st.expander(f"#{rid}｜项目 {pid}｜{proj_title}｜{lp}｜{ts}", expanded=False):
            det = cur.execute("SELECT output_text, src_path FROM trans_ext WHERE id=?", (rid,)).fetchone()
            tgt_full, src_path = det or ("", "")
            st.code(prev or "", language="text")
            st.text_area("译全文", tgt_full or "", height=220, key=f"hist_full_{rid}")

            try:
                src_full = read_source_file(src_path) if src_path else ""
            except Exception:
                src_full = ""

            with st.expander("原文预览(若上传了源文件)", expanded=False):
                st.text_area("原文全文", src_full or "(未保存/未上传源文件)", height=160, key=f"hist_src_{rid}")

            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                if st.button("➕ 添加进语料库", key=f"hist_add_corpus_{rid}"):
                    if not src_full and not tgt_full:
                        st.warning("原文和译文都为空，无法写入语料库。")
                    else:
                        cur.execute(
                            """
                                INSERT INTO corpus (title, project_id, lang_pair, src_text, tgt_text, note, domain, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                            """,
                            (
                                f"{proj_title} · history#{rid}",
                                pid,
                                lp or "",
                                src_full or None,
                                tgt_full or "",
                                f"from trans_ext#{rid}",
                                proj_domain or "",
                            ),
                        )
                        conn.commit()
                        st.success("✅ 已写入语料库")

                if st.button("➕ 添加并重建索引", key=f"hist_add_corpus_rebuild_{rid}"):
                    if not src_full and not tgt_full:
                        st.warning("原文和译文都为空，无法写入语料库。")
                    else:
                        cur.execute(
                            """
                                INSERT INTO corpus (title, project_id, lang_pair, src_text, tgt_text, note, domain, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                            """,
                            (
                                f"{proj_title} · history#{rid}",
                                pid,
                                lp or "",
                                src_full or None,
                                tgt_full or "",
                                f"from trans_ext#{rid}",
                                proj_domain or "",
                            ),
                        )
                        conn.commit()

                        res_idx = rebuild_project_semantic_index(cur, pid, split_fn=_split_pair_for_index)
                        if res_idx.get("ok"):
                            st.success(
                                f"✅ 已写入语料库并重建索引: 新增 {res_idx['added']} 条, 总量 {res_idx['total']} 条"
                            )
                        else:
                            st.warning(
                                f"已写入语料库，但重建索引失败: {res_idx.get('msg','未知错误')}"
                            )

            with c2:
                if st.button("🧠 提取术语", key=f"hist_extract_terms_{rid}"):
                    ak, model = get_deepseek()
                    if not ak:
                        st.info("未检测到 DeepSeek Key，将仅使用语料库同款模型进行抽取。")

                    big = ((src_full or "") + "\n" + (tgt_full or "")).strip()
                    res = ds_extract_terms(
                        big,
                        ak,
                        model,
                        src_lang="zh",
                        tgt_lang="en",
                        prefer_corpus_model=False,
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
                            cur.execute(
                                """
                                    INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    o.get("source_term") or "",
                                    (o.get("target_term") or None),
                                    domain_val,
                                    pid,
                                    strategy_val,
                                    (o.get("example") or None),
                                ),
                            )
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

            with c3:
                if st.button("⬇️ CSV 对照", key=f"hist_dl_bicsv_btn_{rid}"):
                    if not src_full:
                        st.warning("找不到原文(未上传源文件).无法生成 CSV 对照")
                    else:
                        try:
                            csv_name = f"bilingual_history_{rid}.csv"
                            csv_bytes = export_csv_bilingual((src_full, tgt_full), filename=csv_name)
                        except TypeError:
                            csv_name = f"bilingual_history_{rid}.csv"
                            csv_bytes = export_csv_bilingual(src_full, tgt_full)
                        st.download_button(
                            "下载 CSV",
                            data=csv_bytes,
                            file_name=csv_name,
                            mime="text/csv",
                            key=f"hist_dl_bicsv_{rid}",
                        )

            with c4:
                if st.button("⬇️ DOCX 对照", key=f"hist_dl_bidocx_btn_{rid}"):
                    if not src_full:
                        st.warning("找不到原文(未上传源文件).无法生成 DOCX 对照")
                    else:
                        try:
                            docx_path = export_docx_bilingual(filename=f"bilingual_history_{rid}.docx")
                            with open(docx_path, "rb") as f:
                                data_docx = f.read()
                        except TypeError:
                            data_docx = export_docx_bilingual(src_full, tgt_full)
                        st.download_button(
                            "下载 DOCX",
                            data=data_docx,
                            file_name=f"bilingual_history_{rid}.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            key=f"hist_dl_bidocx_{rid}",
                        )

            with c5:
                with st.expander("🗑 删除本条历史(不可恢复)", expanded=False):
                    st.warning("此操作将永久删除该条 trans_ext 记录(不影响已写入语料库/术语表的数据)。")
                    ok = st.checkbox(f"我确认删除 #{rid}", key=f"hist_del_ck_{rid}")
                    if st.button("确认删除", key=f"hist_del_btn_{rid}") and ok:
                        cur.execute("DELETE FROM trans_ext WHERE id=?", (rid,))
                        conn.commit()
                        st.success("已删除.请刷新页面查看结果。")
                        st.stop()

            st.download_button(
                "下载译文 (TXT)",
                tgt_full or "",
                file_name=f"history_{rid}.txt",
                mime="text/plain",
                key=f"hist_dl_txt_{rid}",
            )
