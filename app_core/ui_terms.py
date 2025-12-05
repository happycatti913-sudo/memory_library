# -*- coding: utf-8 -*-
import re
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st

from app_core.config import make_sk
from app_core.database import _has_col
from app_core.file_ops import export_csv_bilingual, export_docx_bilingual, read_source_file
from app_core.term_extraction import ds_extract_terms
from app_core.translation_ops import get_deepseek
from app_core.ui_common import render_table


def render_term_management(st, cur, conn, base_dir, key_prefix="term"):
    sk = make_sk(key_prefix)

    st.subheader("📘 术语库管理")
    section_labels = [
        "查询与编辑",
        "批量导入 CSV",
        "统计与导出",
        "快速搜索",
        "批量挂接项目",
        "从历史提取术语",
        "分类管理",
    ]
    section_choice = st.sidebar.selectbox("📘 术语库分支", section_labels, key=sk("section"))
    section_containers = [st.container() for _ in section_labels]

    # —— 查询与编辑
    if section_choice == section_labels[0]:
        with section_containers[0]:
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
    elif section_choice == section_labels[1]:
        with section_containers[1]:
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
    elif section_choice == section_labels[2]:
        with section_containers[2]:
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
    elif section_choice == section_labels[3]:
        with section_containers[3]:
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
    elif section_choice == section_labels[4]:
        with section_containers[4]:
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
    elif section_choice == section_labels[5]:
        with section_containers[5]:
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
                            res = ds_extract_terms(big, ak, model, src_lang="zh", tgt_lang="en")
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
                            res = ds_extract_terms(big, ak, model, src_lang="zh", tgt_lang="en")
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
    elif section_choice == section_labels[6]:
        with section_containers[6]:
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
    
