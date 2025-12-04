# -*- coding: utf-8 -*-
"""语料库管理 UI。"""
from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from .config import make_sk
from .corpus_ops import import_corpus_from_upload
from .projects import (
    get_project_fewshot_enabled,
    get_project_ref_ids,
    set_project_fewshot_enabled,
)
from .semantic_index import _load_index, rebuild_project_semantic_index
from .semantic_ops import align_semantic, semantic_retrieve
from .text_utils import (
    _split_pair_for_index,
    extract_pairs_from_docx_table,
    read_docx_tables_info,
    read_docx_text,
    read_pdf_text,
    read_txt,
    split_sen,
)


def split_sents(text: str, lang: str = "auto"):
    return split_sen(text, lang=lang)


def render_corpus_manager(st, cur, conn, pid_prefix="corpus"):
    st.header("📚 语料库管理")
    sk = make_sk(pid_prefix)

    sub = st.tabs(["新建语料", "浏览/检索", "使用与导出"])

    with sub[0]:
        st.subheader("📥 上传 / 对齐 / 入库")

        colA, colB = st.columns(2)
        with colA:
            one_file = st.file_uploader(
                "① 单个文件(DOCX 表格对照 / 单语 DOCX/TXT/PDF)",
                type=["docx", "txt", "pdf"],
                key="up_one",
            )
        with colB:
            two_src = st.file_uploader(
                "② 原文文件(可选:与 ③ 搭配做对齐)",
                type=["docx", "txt", "csv", "pdf"],
                key="up_src",
            )
            two_tgt = st.file_uploader(
                "③ 译文文件(可选:与 ② 搭配做对齐)",
                type=["docx", "txt", "csv", "pdf"],
                key="up_tgt",
            )

        st.divider()
        meta1, meta2, meta3 = st.columns([2, 1, 1])
        with meta1:
            title = st.text_input("语料标题", value="未命名语料")
        with meta2:
            lp = st.selectbox("方向", ["自动", "中译英", "英译中"])
        with meta3:
            pid_val = st.text_input("项目ID(可留空)")
        pid = int(pid_val) if pid_val.strip().isdigit() else None

        pairs = []
        src_text = tgt_text = ""
        preview_df = None

        if one_file is not None and (two_src is None and two_tgt is None):
            ext = (one_file.name.split(".")[-1] or "").lower()
            bio = io.BytesIO(one_file.getvalue())

            if ext == "docx":
                tables = read_docx_tables_info(io.BytesIO(bio.getvalue()))
                if tables:
                    st.caption("检测到 DOCX 表格，优先作为双语对照导入。")
                    pairs = extract_pairs_from_docx_table(
                        io.BytesIO(bio.getvalue()),
                        table_index=0,
                        src_col=0,
                        tgt_col=1,
                    )
                else:
                    src_text = read_docx_text(io.BytesIO(bio.getvalue()))

            elif ext == "txt":
                src_text = read_txt(bio)

            elif ext == "pdf":
                src_text = read_pdf_text(io.BytesIO(bio.getvalue()))

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

        if pairs:
            st.success(f"解析到 {len(pairs)} 对(DOCX 表格)")
            preview_df = pd.DataFrame(pairs[:200], columns=["源句", "目标句"])

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

        elif src_text and not tgt_text:
            sents_src = split_sents(src_text, "zh" if lp.startswith("中") else "auto")
            st.info(f"检测到单语文本，共 {len(sents_src)} 句。将以单语语料写入(译文为空)。")
            preview_df = pd.DataFrame([{ "源句": s, "目标句": "" } for s in sents_src[:200]])

        if preview_df is not None:
            st.dataframe(preview_df, width='stretch')

        c_imp, c_opt, c_build = st.columns([1, 1, 1])
        do_import = c_imp.button("📥 写入语料库", type="primary", key="write_pairs_btn")
        do_build_opt = c_opt.checkbox("导入后立即重建索引", value=True, key="build_vec_opt")
        only_build_now = c_build.button("🧠 仅重建索引(不导入)", key="only_build")

        st.caption("提示: 索引也可以稍后在“使用与索引 / 导出”页的 C 区统一重建。")

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
                pid=pid,
                title=title,
                lp=lp,
                pairs=pairs,
                src_text=src_text,
                tgt_text=tgt_text,
                default_title=default_title,
                build_after_import=do_build_opt,
            )

        if only_build_now:
            if pid:
                res_idx = rebuild_project_semantic_index(cur, pid, split_fn=_split_pair_for_index)
                if res_idx.get("ok"):
                    st.success(f"🧠 索引已重建: 新增 {res_idx['added']}，总量 {res_idx['total']}。")
                else:
                    st.error(f"重建失败: {res_idx.get('msg','未知错误')}")
            else:
                st.warning("请先在上方填写有效的项目ID，再重建索引。")

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
                        (rid,),
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
                                key=sk(f"cor_txt_dl_{rid}"),
                            )
                    with c3:
                        if st.button("导出CSV(中英对照)", key=sk(f"cor_csv_{rid}")):
                            df_out = pd.DataFrame([{ "source": src_all, "target": tgt_all }])
                            csv_data = df_out.to_csv(index=False)
                            st.download_button(
                                "下载CSV",
                                csv_data,
                                file_name=f"corpus_{rid}.csv",
                                mime="text/csv",
                                key=sk(f"cor_csv_dl_{rid}"),
                            )
                    with c4:
                        if st.button("删除", key=sk(f"del_{rid}")):
                            cur.execute("DELETE FROM corpus WHERE id=?", (rid,))
                            conn.commit()
                            st.warning("🗑️ 已删除，刷新后生效")
                            st.rerun()

    with sub[2]:
        st.subheader("🧩 使用与索引 / 导出")

        proj_rows = cur.execute(
            "SELECT id, IFNULL(title,'(未命名)') FROM items WHERE COALESCE(type,'')='project' ORDER BY id DESC LIMIT 200"
        ).fetchall()
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
                    ids,
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
                            key=sk(f"export_refs_txt_dl_{target_pid}"),
                        )
            else:
                st.info("还没有选择任何参考语料，可以在“浏览/检索”中勾选后再来。")

        st.markdown("---")

        if not target_pid:
            st.info("选择目标项目后才能配置 Few-shot 注入开关。")
            use_fs = False
        else:
            curr_state = get_project_fewshot_enabled(target_pid)
            use_fs = st.checkbox(
                "翻译时自动注入这些参考语料作为 Few-shot 提示",
                value=curr_state,
                key=sk(f"use_fs_{target_pid}"),
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

        st.subheader("🧠 语义索引管理 & 召回测试")

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
                        st.success(f"索引已更新: 新增 {res_idx['added']}，总量 {res_idx['total']}。")
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
                        sc, m, txt = row[:3]
                        st.write(f"**{sc:.3f}** | {m.get('title','')} | {m.get('lang_pair','')}")
                        st.code(txt, language="text")
                        st.markdown("---")
            else:
                st.warning("请先选择项目并输入要检索的内容。")
