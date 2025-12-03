# -*- coding: utf-8 -*-
"""语料导入与索引更新工具。"""

from __future__ import annotations

import streamlit as st

from app_core.semantic_index import rebuild_project_semantic_index
from app_core.text_utils import _split_pair_for_index, split_sents


def import_corpus_from_upload(
    st,
    cur,
    conn,
    *,
    pid: int | None,
    title: str | None,
    lp: str,
    pairs,
    src_text: str | None,
    tgt_text: str | None,
    default_title: str = "",
    build_after_import: bool = False,
):
    """统一的“上传语料→写入数据库→可选重建索引”流程。"""
    base_title = (title or default_title or "").strip() or "未命名语料"

    def normalize_pairs_to2(pairs_in):
        if not pairs_in:
            return []
        if len(pairs_in[0]) == 3:
            return [(s, t) for (s, t, _) in pairs_in]
        return pairs_in

    if pairs:
        pairs2 = normalize_pairs_to2(pairs)
        ins = 0
        for s, t in pairs2:
            s = (s or "").strip()
            t = (t or "").strip()
            if not (s or t):
                continue
            cur.execute(
                """
                INSERT INTO corpus(title, project_id, lang_pair, src_text, tgt_text, note, created_at)
                VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
                """,
                (
                    base_title,
                    pid,
                    lp,
                    s or None,
                    t or None,
                    "auto-import",
                ),
            )
            ins += 1
        conn.commit()
        st.success(f"✅ 已写入语料库 {ins} 条。")

        if build_after_import and pid:
            res_idx = rebuild_project_semantic_index(cur, pid, split_fn=_split_pair_for_index)
            if res_idx.get("ok"):
                st.success(
                    f"🧠 向量索引已更新: 新增 {res_idx['added']}，总量 {res_idx['total']}。"
                )
            else:
                st.warning(f"索引未更新: {res_idx.get('msg','未知错误')}")

        return

    if src_text and not tgt_text:
        lang_hint = "zh" if (lp or "").startswith("中") else "en"
        sents = split_sents(src_text, lang_hint)
        ins = 0
        for s in sents:
            s = (s or "").strip()
            if not s:
                continue
            cur.execute(
                """
                INSERT INTO corpus(title, project_id, lang_pair, src_text, tgt_text, note, created_at)
                VALUES (?, ?, ?, ?, NULL, ?, datetime('now'))
                """,
                (
                    base_title,
                    pid,
                    lp,
                    s,
                    "mono",
                ),
            )
            ins += 1
        conn.commit()
        st.success(f"✅ 已写语料库 {ins} 条。")

        if build_after_import and pid:
            res_idx = rebuild_project_semantic_index(cur, pid, split_fn=_split_pair_for_index)
            if res_idx.get("ok"):
                st.success(
                    f"🧠 向量索引已更新: 新增 {res_idx['added']}，总量 {res_idx['total']}。"
                )
            else:
                st.warning(f"索引未更新: {res_idx.get('msg','未知错误')}")

        return

    st.info("未检测到可写入的语料内容。")
