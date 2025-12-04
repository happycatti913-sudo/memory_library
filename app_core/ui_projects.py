# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st

from app_core.config import highlight_terms
from app_core.file_ops import read_source_file
from app_core.projects import (
    cleanup_project_files,
    ensure_legacy_file_record,
    fetch_project_files,
    get_project_fewshot_examples,
    register_project_file,
    remove_project_file,
)
from app_core.semantic_ops import semantic_consistency_report
from app_core.text_utils import split_paragraphs, split_sents
from app_core.term_ops import get_terms_for_project
from app_core.translation_ops import get_deepseek, translate_block_with_kb


def run_project_translation_ui(pid, project_title, src_path, conn, cur):
    """执行翻译 UI + 逻辑（从 1127.py 抽取）。"""

    st.subheader(f"📘 项目：{project_title}")
    st.info("打工不易，牛马哭泣。")

    selected_src_path = src_path

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
    st.code(repr(src_text[:400]))  # 看字符串里有没有 '\n'
    st.write({"len": len(src_text), "nl": src_text.count("\n"), "cr": src_text.count("\r")})
    st.write({"preview_lines": src_text.splitlines()[:3]})

    # 用统一的 split_paragraphs 做切分
    blocks = split_paragraphs(src_text)
    if not blocks:
        st.error("源文件内容为空，或未识别到有效段落")
        st.stop()

    st.info(f"按段落切分，共 {len(blocks)} 段，开始翻译…")

    lang_pair_val = st.session_state.get(f"lang_{pid}", "中译英")
    use_semantic = bool(st.session_state.get(f"use_sem_{pid}", True))
    scope_val = st.session_state.get(f"scope_{pid}", st.session_state.get("scope_newproj", "project"))

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

        res = translate_block_with_kb(
            cur=cur,
            project_id=pid,
            block_text=blk,
            lang_pair=lang_pair_val,
            ak=ak,
            model=model,
            use_semantic=use_semantic,
            scope=scope_val,
            fewshot_examples=fewshot_examples,
        )

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
    blocks_src_safe = list(blocks if 'blocks' in locals() else [])
    if len(blocks_src_safe) != len(all_results_safe):
        n = min(len(blocks_src_safe), len(all_results_safe))
        blocks_src_safe = blocks_src_safe[:n]
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
    proj_title = project_title or f"project_{pid}"
    st.download_button(
        "⬇️ 下载翻译结果 (TXT)",
        final_text or "",
        file_name=f"{proj_title}_翻译结果.txt",
        mime="text/plain",
        key=f"dl_txt_{pid}"
    )

    # 写入历史 trans_ext
    try:
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
            selected_src_path,  # 源文件路径(可为空)
            lang_pair_val,   # 语对
            "标准模式",      # 模式
            final_text,      # 输出文本(最终译文)
            None,            # 统计JSON(占位)
            seg_count,       # 段数(修复:不用 blocks_src_safe)
            None             # 术语命中数(占位.可后续填真实值)
        ))
        conn.commit()
        st.success("📝 已写入翻译历史")
    except Exception as e:
        st.warning(f"写入翻译历史失败: {e}")


# ====== Tab1 UI ======

def render_project_tab(st, cur, conn, base_dir, use_semantic=True):
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
                key="new_proj_prompt",
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
        return

    # 显示项目列表
    for idx, row in enumerate(rows):
        pid, title, tags, src_path, created_at, scene, prompt_text, mode, trans_type = row
        st.markdown(f"### #{pid}｜{title}")

        col_meta, col_actions = st.columns([3, 2])
        with col_meta:
            st.caption(f"标签: {tags} ｜ 场合: {scene} ｜ 创建: {created_at}")
            st.caption(f"模式: {mode or translation_mode} ｜ 翻译方式: {trans_type or translation_type}")
            st.text_area("项目 Prompt", prompt_text or "(未设置)", height=80, key=f"prompt_{pid}")

        with col_actions:
            # 几个快速设置
            st.checkbox("启用语义召回参考", value=True, key=f"use_sem_{pid}")
            st.text_input("语义召回范围(scope)", st.session_state.get("scope_newproj", "project"), key=f"scope_{pid}")
            st.selectbox("翻译方向", ["中译英", "英译中"], index=0, key=f"lang_{pid}")

        with st.expander("📎 源文件管理"):
            file_records = fetch_project_files(cur, pid)
            selected_src_path = st.selectbox(
                "选择已上传文件作为源文件",
                [r["path"] for r in file_records] if file_records else [],
                key=f"sel_src_{pid}",
            ) if file_records else None

            # 兼容旧数据：若 item_ext 为空但 items.body 有文件内容，则补录
            ensure_legacy_file_record(cur, conn, pid, file_records)

            up_files = st.file_uploader("上传新文件", accept_multiple_files=True, key=f"upload_{pid}")
            if up_files:
                file_records, saved_paths = register_project_file(cur, conn, pid, up_files, base_dir)
                saved = len(saved_paths)

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

            # 删除项目
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
                cur=cur,
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
            draft = st.session_state.get(f"workspace_draft_{pid}", [])
            terms = st.session_state.get(f"workspace_terms_{pid}", [])

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
                        highlighted = highlight_terms(new_trg, terms) if terms else new_trg
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

    st.caption("提示:如需批量删除，可在上方勾选多个项目。")
