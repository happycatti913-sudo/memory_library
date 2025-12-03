# -*- coding: utf-8 -*-
"""索引管理相关的 Streamlit UI 组件。"""

import streamlit as st
import pandas as pd

from app_core.semantic_index import (
    _load_index,
    _load_index_domain,
    build_project_vector_index,
    build_strategy_index_for_domain,
    rebuild_project_semantic_index,
)
from app_core.text_utils import _split_pair_for_index


def render_index_manager(st, conn, cur):
    """
    🧠 统一的索引管理页面:
      - 按项目查看: 语料条数 / 其中来自历史的条数 / 索引条数拆分
      - 一键重建当前项目索引
      - (可选) 批量重建
    """
    st.title("🧠 语义索引管理")

    # === 1. 项目列表 + 基本统计 ===
    st.markdown("#### 项目概览")

    rows = cur.execute(
        """
        SELECT i.id,
               IFNULL(i.title,''),
               IFNULL(i.domain,''),
               COUNT(DISTINCT c.id)                                           AS corpus_cnt,
               SUM(CASE WHEN c.note LIKE 'from trans_ext%%' THEN 1 ELSE 0 END) AS hist_cnt
        FROM items i
        LEFT JOIN corpus c ON c.project_id = i.id
        GROUP BY i.id, i.title, i.domain
        ORDER BY i.id DESC
        LIMIT 500;
        """
    ).fetchall()

    if not rows:
        st.info("当前还没有任何项目或语料。请先在项目管理/语料库中添加内容。")
        return

    df_proj = pd.DataFrame(
        [
            {
                "项目ID": r[0],
                "项目名称": r[1],
                "领域": r[2],
                "语料条数": r[3],
                "其中来自翻译历史": r[4] or 0,
            }
            for r in rows
        ]
    )

    st.dataframe(df_proj, use_container_width=True)

    # === 2. 选择一个项目，查看索引状态 ===
    st.markdown("#### 单项目索引状态 & 操作")

    proj_options = {f"[{r[0]}] {r[1] or '(未命名)'}": r[0] for r in rows}
    proj_label = st.selectbox(
        "选择要查看/重建索引的项目",
        ["(请选择)"] + list(proj_options.keys()),
    )
    pid_sel = proj_options.get(proj_label)

    if not pid_sel:
        st.info("请选择一个项目以查看索引详情。")
        return

    # 2.1 统计当前索引中的 source=corpus / history 数量
    idx_total = idx_corpus = idx_hist = idx_other = 0
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
    except Exception as e:
        st.warning(f"读取索引文件失败: {e}")

    st.write(
        f"- 当前索引条数: **{idx_total}** 条\n"
        f"- 其中来自语料库(corpus): **{idx_corpus}** 条\n"
        f"- 其中来自翻译历史(history): **{idx_hist}** 条\n"
        f"- 其他/未知来源: **{idx_other}** 条"
    )

    # 2.2 当前项目在 DB 中的语料统计，和上面的 df_proj 对应
    row_sel = [r for r in rows if r[0] == pid_sel][0]
    st.write(
        f"- 数据库中语料条数: **{row_sel[3]}** 条 "
        f"(其中来自翻译历史: **{row_sel[4] or 0}** 条)"
    )

    # === 3. 操作区: 重建索引 / 批量重建 ===
    c1, c2 = st.columns(2)

    with c1:
        if st.button("🔁 重建当前项目索引", type="primary", key=f"rebuild_idx_{pid_sel}"):
            res = rebuild_project_semantic_index(cur, pid_sel, split_fn=_split_pair_for_index)
            if res.get("ok"):
                st.success(
                    f"索引已重建: 新增 {res['added']} 条, 总量 {res['total']} 条。"
                )
            else:
                st.error(f"重建失败: {res.get('msg','未知错误')}")

    with c2:
        if st.button("⚠ 批量重建上表列出的全部项目索引", key="rebuild_all_idx"):
            ok_cnt = fail_cnt = 0
            for r in rows:
                pid = r[0]
                res = rebuild_project_semantic_index(cur, pid, split_fn=_split_pair_for_index)
                if res.get("ok"):
                    ok_cnt += 1
                else:
                    fail_cnt += 1
            st.success(f"批量重建完成: 成功 {ok_cnt} 个项目, 失败 {fail_cnt} 个项目。")


def render_index_manager_by_domain(st, conn, cur):
    """按领域管理语义索引与策略索引。"""
    st.title("🧠 按领域管理索引与策略库")

    domains = set()
    try:
        rows = cur.execute("SELECT DISTINCT IFNULL(domain,'未分类') FROM items WHERE COALESCE(type,'')='project'").fetchall()
        for (d,) in rows:
            if d and d.strip():
                domains.add(d.strip())
            else:
                domains.add("未分类")

        rows = cur.execute("SELECT DISTINCT IFNULL(domain,'未分类') FROM corpus").fetchall()
        for (d,) in rows:
            if d and d.strip():
                domains.add(d.strip())
            else:
                domains.add("未分类")

        rows = cur.execute("SELECT DISTINCT IFNULL(domain,'未分类') FROM strategy_texts").fetchall()
        for (d,) in rows:
            if d and d.strip():
                domains.add(d.strip())
            else:
                domains.add("未分类")
    except Exception:
        pass

    if not domains:
        st.info("当前尚未设置任何领域(domain)。请先在项目或语料中设置领域。")
        return

    domains_list = sorted(domains)
    dom_sel = st.selectbox("选择要管理的领域", domains_list)
    dom_key = (dom_sel or "").strip() or "未分类"

    st.markdown(f"### 当前领域: `{dom_key}`")

    # 2) 统计该领域下的项目 & 语料 & 索引情况
    # 2.1 项目列表
    proj_rows = cur.execute(
        "SELECT id, title FROM items WHERE IFNULL(domain,'未分类') = ? ORDER BY id ASC",
        (dom_key,),
    ).fetchall()
    proj_ids = [pid for (pid, _) in proj_rows]

    # 2.2 语料条数(如果 corpus 有 domain 字段)
    corpus_cnt = None
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(corpus)").fetchall()]
        if "domain" in cols:
            corpus_cnt = cur.execute(
                "SELECT COUNT(*) FROM corpus WHERE IFNULL(domain,'未分类') = ?",
                (dom_key,),
            ).fetchone()[0]
    except Exception:
        corpus_cnt = None

    # 2.3 双语索引条数 = 该领域所有项目索引的 mapping 长度之和
    idx_bilingual_total = 0
    for pid in proj_ids:
        try:
            mode, index, mapping, vecs = _load_index(int(pid))
        except Exception:
            continue
        if isinstance(mapping, list):
            idx_bilingual_total += len(mapping)

    # 2.4 策略文本数量 & 策略索引条数
    try:
        strategy_cnt = cur.execute(
            "SELECT COUNT(*) FROM strategy_texts WHERE IFNULL(domain,'未分类') = ?",
            (dom_key,),
        ).fetchone()[0]
    except Exception:
        strategy_cnt = 0

    mode_s, index_s, mapping_s, vecs_s = _load_index_domain(dom_key, "strategy")
    if isinstance(mapping_s, list):
        idx_strategy_total = len(mapping_s)
    else:
        idx_strategy_total = 0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📘 双语对照索引(例句库)")
        st.write(f"- 该领域下项目数: **{len(proj_ids)}**")
        if corpus_cnt is not None:
            st.write(f"- 语料库中双语条目(按 domain 计): **{corpus_cnt}**")
        st.write(f"- 已建立索引的句对条数(合计): **{idx_bilingual_total}**")

        if proj_ids:
            if st.button("🔁 重建该领域所有项目的【双语对照】索引", key=f"rebuild_bi_{dom_key}"):
                added_sum = 0
                total_sum = 0
                for pid in proj_ids:
                    try:
                        res = build_project_vector_index(cur, int(pid), split_fn=_split_pair_for_index)
                        added_sum += res.get("added", 0)
                        total_sum = res.get("total", total_sum)
                    except Exception as e:
                        st.warning(f"项目 {pid} 重建索引时出错: {e}")
                st.success(
                    f"已重建该领域所有项目索引。"
                    f"新增句对: {added_sum}，最后一个项目返回的索引总量: {total_sum}"
                )
        else:
            st.info("该领域下暂时没有任何项目。")

    with c2:
        st.markdown("#### 📝 翻译策略索引(strategy)")
        st.write(f"- 该领域下策略文本条数: **{strategy_cnt}**")
        st.write(f"- 已建立索引的策略向量条数: **{idx_strategy_total}**")

        if st.button("🔁 重建该领域的【翻译策略】索引", key=f"rebuild_strategy_{dom_key}"):
            try:
                res = build_strategy_index_for_domain(dom_key)
                st.success(
                    f"已重建策略索引。新增策略段落: {res.get('added', 0)}，"
                    f"索引总量: {res.get('total', 0)}"
                )
            except Exception as e:
                st.error(f"重建策略索引时出错: {e}")

    st.markdown("---")
    with st.expander("🔍 查看该领域下的项目列表", expanded=False):
        if proj_rows:
            for pid, title in proj_rows:
                st.write(f"- 项目 {pid}: {title}")
        else:
            st.write("暂无项目。")

    with st.expander("🔍 查看该领域下的策略文本(前几条)", expanded=False):
        try:
            rows = cur.execute(
                "SELECT id, title, substr(content,1,200) FROM strategy_texts "
                "WHERE IFNULL(domain,'未分类') = ? ORDER BY id DESC LIMIT 20",
                (dom_key,),
            ).fetchall()
            if not rows:
                st.write("暂无策略文本。")
            else:
                for rid, title, content_prev in rows:
                    st.write(f"- #{rid} {title}: {content_prev}…")
        except Exception as e:
            st.write(f"读取策略文本出错: {e}")

