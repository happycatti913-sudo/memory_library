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
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
import altair as alt


# 让同目录下的 kb_dynamic.py 可被导入(如果存在)
sys.path.append(os.path.dirname(__file__))

# ======== 基本路径设置 ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kb.db")

PROJECT_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(PROJECT_DIR, exist_ok=True)

# 统一的语义索引根目录: semantic_index/{project_id}/...
SEM_INDEX_ROOT = os.path.join(BASE_DIR, "semantic_index")
os.makedirs(SEM_INDEX_ROOT, exist_ok=True)

def _norm_domain_key(raw: str | None) -> str:
    """
    把数据库里的 domain 字段转成适合作为文件夹名的 key：
    - None/空 → "未分类"
    - 去掉首尾空格
    - 替换掉不适合作为路径的字符(Windows 下的保留字符)
    """
    s = (raw or "").strip()
    if not s:
        s = "未分类"
    for ch in r'\\/:"*?<>|':
        s = s.replace(ch, "_")
    return s


# ---------- 语义索引路径:按“领域 → 类型”归类 ----------
def _index_paths(project_id: int):
    """
    统一的语义索引路径(按“领域/类型”归类):

        BASE_DIR / semantic_index / {domain_key} / bilingual / index.faiss
                                                       / mapping.json
                                                       / vectors.npy

    目前仅支持双语对照(bilingual)索引;
    未来如果增加翻译策略(strategy), 可以在这里扩展 kb_type 参数.
    """
    # 尝试根据项目推断领域; 拿不到时归入“未分类”
    domain_raw = None
    try:
        if "cur" in globals():
            row = cur.execute(
                "SELECT IFNULL(domain,'') FROM items WHERE id=?",
                (int(project_id),)
            ).fetchone()
            if row:
                domain_raw = (row[0] or "").strip()
    except Exception:
        domain_raw = None

    domain_key = _norm_domain_key(domain_raw)
    kb_type = "bilingual"

    base_dir = os.path.join(BASE_DIR, "semantic_index", domain_key, kb_type)
    os.makedirs(base_dir, exist_ok=True)

    idx_path = os.path.join(base_dir, "index.faiss")
    map_path = os.path.join(base_dir, "mapping.json")
    vec_path = os.path.join(base_dir, "vectors.npy")

    return idx_path, map_path, vec_path


def _project_domain(pid: int | None) -> str | None:
    """安全获取项目的领域标签。"""
    if not pid:
        return None
    try:
        row = cur.execute("SELECT IFNULL(domain,'') FROM items WHERE id=?", (int(pid),)).fetchone()
        dom = (row[0] if row else "").strip()
        return dom or None
    except Exception:
        return None


def dedup_terms_against_db(
    cur,
    terms: list[dict],
    project_id: int | None,
):
    """
    按 (source_term, domain) 去重，过滤已存在或本次重复的术语。

    - 与 term_ext 中同一项目或全局术语重复时跳过。
    - domain 为空时按空串参与去重，确保同源术语仅保留一次。
    返回 (filtered, skipped)。
    """

    if not terms:
        return [], []

    try:
        rows = cur.execute(
            """
            SELECT source_term, domain
            FROM term_ext
            WHERE project_id IS NULL OR project_id = ?
            """,
            (project_id if project_id is not None else -1,),
        ).fetchall()
    except Exception:
        rows = []

    existing = {
        ((s or "").strip().lower(), (d or "").strip().lower())
        for s, d in rows
        if (s or "").strip()
    }

    filtered, skipped = [], []
    for item in terms:
        src = (item.get("source_term") or "").strip()
        dom = (item.get("domain") or "").strip()
        if not src:
            skipped.append(item)
            continue
        key = (src.lower(), dom.lower())
        if key in existing:
            skipped.append(item)
            continue
        existing.add(key)
        filtered.append(item | {"source_term": src, "domain": dom})
    return filtered, skipped

# ======== 轻量日志机制 ========
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def log_event(level: str, message: str, **extra):
    """
    轻量日志记录:
        level  : "INFO" / "WARNING" / "ERROR"
        message: 简短描述
        extra  : 可选的结构化字段, 会一起写入 JSON
    写入路径: BASE_DIR/logs/app.log
    """
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "time": ts,
            "level": (level or "INFO").upper(),
            "message": str(message),
        }
        if extra:
            record["extra"] = extra
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # 日志本身永远不能炸应用，静默失败
        pass

# ==== third-party ====
try:
    from docx import Document  # 在需要处仍会 try/except
except Exception:
    Document = None

# ========== 页面设置 ==========
st.set_page_config(page_title="个人翻译知识库管理系统3.0", layout="wide")

# ========== kb_dynamic (可选) ==========
KBEmbedder = None
recommend_for_segment = None
build_prompt_strict = None
build_prompt_soft = None
try:
    from kb_dynamic import (
        KBEmbedder as _KBEmbedder,
        recommend_for_segment as _recommend_for_segment,
        build_prompt_strict as _build_prompt_strict,
        build_prompt_soft as _build_prompt_soft,
    )
    KBEmbedder = _KBEmbedder
    recommend_for_segment = _recommend_for_segment
    build_prompt_strict = _build_prompt_strict
    build_prompt_soft = _build_prompt_soft
except Exception:
    pass  # 允许缺失;动态术语推荐功能将自动降级

# ========== 工具函数 ==========
def make_sk(prefix: str):
    """返回一个带有前缀的 key 生成器"""
    return lambda name, id=None: f"{prefix}_{name}_{id}" if id else f"{prefix}_{name}"
# 全局默认 key 生成器(替代被删除的计数器版 sk)
sk = make_sk("global")

def render_table(df, *, key=None, hide_index=True, editable=False):
    """
    统一渲染表格(对旧/新 Streamlit 都安全):
    - editable=False: 只读(用 data_editor disabled=True 以保留 key)
    - editable=True : 可编辑
    - 不再传 width 参数.避免 'str' as int 的报错
    """
    try:
        if editable:
            return st.data_editor(
                df,
                hide_index=hide_index,
                key=key,
            )
        if key is not None:
            return st.data_editor(
                df,
                hide_index=hide_index,
                disabled=True,
                key=key,
            )
        return st.dataframe(df, hide_index=hide_index)
    except TypeError:
        return st.data_editor(
            df,
            hide_index=hide_index,
            disabled=not editable,
            key=key,
        )
# ========== 文本高亮函数 ==========
def highlight_terms(text: str, term_pairs: list):
    """高亮术语，term_pairs = [(src, tgt), ..]"""
    if not term_pairs:
        return text

    import re
    safe = text

    for s, t in term_pairs:
        if not s:
            continue
        # 对 source term 做高亮（黄色）
        safe = re.sub(
            re.escape(s),
            fr"<span style='background: #fff3b0'>{s}</span>",
            safe,
            flags=re.IGNORECASE
        )
        # 对 target term 也高亮（淡绿）
        if t:
            safe = re.sub(
                re.escape(t),
                fr"<span style='background: #d4f6d4'>{t}</span>",
                safe,
                flags=re.IGNORECASE
            )

    return safe

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

    import pandas as pd

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
            res = rebuild_project_semantic_index(pid_sel)
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
                res = rebuild_project_semantic_index(pid)
                if res.get("ok"):
                    ok_cnt += 1
                else:
                    fail_cnt += 1
            st.success(f"批量重建完成: 成功 {ok_cnt} 个项目, 失败 {fail_cnt} 个项目。")
# ======= 获取某条历史记录对应的原文(优先 items.body.兜底 src_path 仅作为标题提示)=======
def get_terms_for_project(cur, pid: int, use_dynamic: bool = True):
    """
    统一术语加载接口（核心主干之一）

    参数:
        cur        : SQLite cursor
        pid        : 当前项目 ID
        use_dynamic: 是否包含“动态术语”（其他项目或全局术语）

    返回:
        term_map, term_meta

        term_map : dict[str, str]
            {源术语: 目标术语}，用于 prompt 注入 / 一致性检查等。

        term_meta: list[dict]
            每条术语的元信息，例如:
            {
                "source_term": "...",
                "target_term": "...",
                "domain": "...",
                "origin": "static" | "dynamic",
                "project_id": 123   # 术语所属项目ID, static 时=当前项目
            }

    说明:
        - 静态术语: term_ext.project_id = 当前项目 pid
        - 动态术语: term_ext.project_id IS NULL 或 <> pid
        - 去重规则: 不区分大小写, (source_term, domain) 相同只保留一条
    """

    # 1) 静态术语（挂接到当前项目的术语）
    rows_static = cur.execute(
        """
        SELECT source_term, target_term, domain
        FROM term_ext
        WHERE project_id = ?
        """,
        (pid,),
    ).fetchall()

    # 2) 动态术语（其他项目 / 全局术语）
    if use_dynamic:
        rows_dynamic = cur.execute(
            """
            SELECT source_term, target_term, domain, project_id
            FROM term_ext
            WHERE project_id IS NULL OR project_id <> ?
            """,
            (pid,),
        ).fetchall()
    else:
        rows_dynamic = []

    # 3) 去重：key = (lower(source), lower(domain))
    #    value = (source_term, target_term, domain, origin, term_project_id)
    dedup: dict[tuple[str, str], tuple[str, str, str, str, int | None]] = {}

    # 静态
    for s, t, d in rows_static:
        if not s:
            continue
        s_raw = (s or "").strip()
        t_raw = (t or "").strip()
        d_raw = (d or "").strip()
        key = (s_raw.lower(), d_raw.lower())
        if key not in dedup:
            dedup[key] = (s_raw, t_raw, d_raw, "static", int(pid))

    # 动态
    for row in rows_dynamic:
        # row 结构: (source_term, target_term, domain, project_id)
        if len(row) == 4:
            s, t, d, pid_term = row
        else:
            # 兼容意外情况
            s, t, d = row[0], row[1], row[2]
            pid_term = None
        if not s:
            continue
        s_raw = (s or "").strip()
        t_raw = (t or "").strip()
        d_raw = (d or "").strip()
        key = (s_raw.lower(), d_raw.lower())
        if key not in dedup:
            dedup[key] = (s_raw, t_raw, d_raw, "dynamic", pid_term if pid_term is not None else None)

    # 4) 拼出 term_map 和 term_meta
    term_map: dict[str, str] = {}
    term_meta: list[dict] = []

    for (_s_lc, _d_lc), (s_raw, t_raw, d_raw, origin, pid_term) in dedup.items():
        if not s_raw:
            continue
        term_map[s_raw] = t_raw
        term_meta.append(
            {
                "source_term": s_raw,
                "target_term": t_raw,
                "domain": d_raw,
                "origin": origin,
                "project_id": pid_term,
            }
        )

    return term_map, term_meta

# ======= 轻量术语候选(中英都可;你后续可换成 DeepSeek 抽取)=======

def register_project_file(cur, conn, project_id, file_name, data_bytes):
    """
    将上传的文件保存到项目目录，并记录在 project_files 表中。
    """
    if not project_id or not data_bytes:
        return None
    safe_name = os.path.basename(file_name) or f"project_{project_id}_file"
    proj_dir = os.path.join(PROJECT_DIR, f"project_{project_id}")
    os.makedirs(proj_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uniq_name = f"{stamp}_{uuid.uuid4().hex[:6]}_{safe_name}"
    full_path = os.path.join(proj_dir, uniq_name)
    with open(full_path, "wb") as f:
        f.write(data_bytes)
    cur.execute(
        """
        INSERT INTO project_files (project_id, file_path, file_name)
        VALUES (?, ?, ?)
        """,
        (project_id, full_path, safe_name),
    )
    conn.commit()
    return full_path


def fetch_project_files(cur, project_id):
    if not project_id:
        return []
    rows = cur.execute(
        """
        SELECT id, IFNULL(file_name,''), IFNULL(file_path,''), IFNULL(uploaded_at,'')
        FROM project_files
        WHERE project_id=?
        ORDER BY id DESC
        """,
        (project_id,),
    ).fetchall()
    items = []
    for fid, name, path, uploaded in rows:
        if not path:
            continue
        display = name or os.path.basename(path) or f"file_{fid}"
        items.append(
            {
                "id": fid,
                "name": display,
                "path": path,
                "uploaded_at": uploaded,
            }
        )
    return items


def ensure_legacy_file_record(cur, conn, project_id, legacy_path):
    """
    旧版仅支持单文件，若检测到 item_ext.src_path，自动同步到 project_files。
    """
    if not (project_id and legacy_path):
        return
    exists = cur.execute(
        "SELECT 1 FROM project_files WHERE project_id=? AND file_path=?",
        (project_id, legacy_path),
    ).fetchone()
    if exists:
        return
    cur.execute(
        """
        INSERT INTO project_files (project_id, file_path, file_name)
        VALUES (?, ?, ?)
        """,
        (project_id, legacy_path, os.path.basename(legacy_path) or None),
    )
    conn.commit()


def remove_project_file(cur, conn, file_id):
    row = cur.execute("SELECT file_path FROM project_files WHERE id=?", (file_id,)).fetchone()
    if row:
        (path,) = row
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    cur.execute("DELETE FROM project_files WHERE id=?", (file_id,))
    conn.commit()


def cleanup_project_files(cur, conn, project_id):
    rows = cur.execute("SELECT file_path FROM project_files WHERE project_id=?", (project_id,)).fetchall()
    for (fp,) in rows:
        if fp and os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass
    cur.execute("DELETE FROM project_files WHERE project_id=?", (project_id,))
    conn.commit()


def _ensure_project_ref_map():
    """
    确保 session_state['corpus_refs'] 为 {project_id: set(ids)} 结构。
    """
    refs = st.session_state.get("corpus_refs")
    if isinstance(refs, dict):
        return refs
    st.session_state["corpus_refs"] = {}
    return st.session_state["corpus_refs"]


def _ensure_project_switch_map():
    """
    确保 session_state['cor_use_ref'] 为 {project_id: bool} 结构。
    """
    switches = st.session_state.get("cor_use_ref")
    if isinstance(switches, dict):
        return switches
    st.session_state["cor_use_ref"] = {}
    return st.session_state["cor_use_ref"]


def get_project_ref_ids(project_id: int | None) -> set[int]:
    if not project_id:
        return set()
    ref_map = _ensure_project_ref_map()
    ref_map.setdefault(project_id, set())
    return ref_map[project_id]


def get_project_fewshot_enabled(project_id: int | None) -> bool:
    if not project_id:
        return False
    switch_map = _ensure_project_switch_map()
    return bool(switch_map.get(project_id, False))


def set_project_fewshot_enabled(project_id: int | None, value: bool):
    if not project_id:
        return
    switch_map = _ensure_project_switch_map()
    switch_map[project_id] = bool(value)


def get_project_fewshot_examples(
    cur,
    project_id: int | None,
    *,
    limit: int | None = 5,
    require_enabled: bool = True,
):
    if not project_id:
        return []
    if require_enabled and not get_project_fewshot_enabled(project_id):
        return []

    ref_ids = list(get_project_ref_ids(project_id))
    if not ref_ids:
        return []

    ref_ids = sorted({int(rid) for rid in ref_ids if str(rid).isdigit()}, reverse=True)
    if limit is not None and len(ref_ids) > limit:
        ref_ids = ref_ids[:limit]

    qmarks = ",".join(["?"] * len(ref_ids))
    rows = cur.execute(
        f"SELECT id, title, IFNULL(src_text,''), IFNULL(tgt_text,'') FROM corpus WHERE id IN ({qmarks})",
        ref_ids,
    ).fetchall()
    order_map = {rid: idx for idx, rid in enumerate(ref_ids)}
    rows.sort(key=lambda r: order_map.get(r[0], len(order_map)))

    examples = []
    for rid, title, src, tgt in rows:
        src_norm = (src or "").strip()
        tgt_norm = (tgt or "").strip()
        if not (src_norm and tgt_norm):
            continue
        examples.append(
            {
                "id": rid,
                "title": title or f"示例#{rid}",
                "src": src_norm,
                "tgt": tgt_norm,
            }
        )
    return examples
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
            thr=0.70
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
def quick_diagnose_vectors(pid: int):
    """
    打印/提示项目向量索引状态.帮助排查“检索为空/维度不匹配/未建索引”等问题。
    """
    try:
        mode, index, mapping, vecs = _load_index(pid)
        if mode == "none":
            # 提示一下当前索引应当所在的领域/路径
            dom = None
            try:
                if "cur" in globals():
                    dom = _get_domain_for_proj(cur, int(pid))  # type: ignore[name-defined]
            except Exception:
                dom = None
            dom_key = _norm_domain_key(dom)
            st.warning(
                f"项目 {pid} 尚未建立向量索引(semantic_index/{dom_key}/bilingual 下无索引文件)。"
            )
            return
        msg = f"索引模式: {mode}; 映射条数: {len(mapping)}"

        if mode == "faiss" and index is not None:
            msg += f"; FAISS ntotal: {index.ntotal}"
        elif vecs is not None:
            msg += f"; NPY shape: {getattr(vecs, 'shape', None)}"
        st.info(msg)

        # 抽样验证映射的 corpus_id 是否都能回查到文本
        bad = 0
        for m in mapping[:10]:
            cid = int(m.get("corpus_id") or -1)
            row = cur.execute("SELECT id FROM corpus WHERE id=?", (cid,)).fetchone()
            if not row:
                bad += 1
        if bad:
            st.warning(f"映射中有 {bad} 条 corpus_id 无法回查.请考虑重建索引。")
    except Exception as e:
        st.error(f"向量诊断异常:{e}")

# ---------- 文件读取工具 ----------
def _lazy_docx():
    try:
        import docx  # python-docx
        return docx
    except Exception:
        return None

def _normalize(t: str) -> str:
    if not t: return ""
    t = t.replace("\xa0", " ").replace("\u200b", "")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()

def read_docx_tables_info(file_like):
    docx = _lazy_docx()
    if not docx: return {}
    try:
        doc = docx.Document(file_like)
    except Exception:
        return {}
    info = {}
    for ti, tbl in enumerate(doc.tables):
        rows = len(tbl.rows)
        cols = len(tbl.columns) if rows else 0
        prev = []
        for r in tbl.rows[: min(6, rows)]:
            prev.append(tuple(_normalize(c.text) for c in r.cells))
        info[ti] = {"rows": rows, "cols": cols, "preview": prev}
    return info

def extract_pairs_from_docx_table(file_like, table_index=0, src_col=0, tgt_col=1,
                                  ffill=True, drop_empty_both=True, dedup=True):
    docx = _lazy_docx()
    if not docx: return []
    try:
        doc = docx.Document(file_like)
    except Exception:
        return []
    if table_index >= len(doc.tables): return []
    tbl = doc.tables[table_index]
    rows = []
    for r in tbl.rows:
        rows.append([_normalize(c.text) for c in r.cells])
    if not rows: return []
    max_cols = max(len(r) for r in rows)
    if src_col >= max_cols or tgt_col >= max_cols: return []

    if ffill:
        for col in (src_col, tgt_col):
            last = ""
            for i in range(len(rows)):
                val = rows[i][col] if col < len(rows[i]) else ""
                if val: last = val
                else: rows[i][col] = last

    pairs = []
    for r in rows:
        s = r[src_col] if src_col < len(r) else ""
        t = r[tgt_col] if tgt_col < len(r) else ""
        s, t = s.strip(), t.strip()
        if drop_empty_both and (not s and not t):
            continue
        pairs.append((s, t))

    if dedup:
        seen, out = set(), []
        for p in pairs:
            if p in seen: continue
            seen.add(p); out.append(p)
        pairs = out
    return pairs

def read_docx_text(file_like) -> str:
    docx = _lazy_docx()
    if not docx: return ""
    try:
        doc = docx.Document(file_like)
    except Exception:
        return ""
    blocks = []
    for p in doc.paragraphs:
        t = _normalize(p.text)
        if t: blocks.append(t)
    # 把表格单元也拼成行.避免漏掉内容
    for tbl in doc.tables:
        for r in tbl.rows:
            line = " ".join(_normalize(c.text) for c in r.cells if _normalize(c.text))
            if line: blocks.append(line)
    return "\n".join(blocks)

def read_txt(file_like_or_bytes) -> str:
    try:
        if hasattr(file_like_or_bytes, "read"):
            data = file_like_or_bytes.read()
        else:
            data = file_like_or_bytes
        if isinstance(data, bytes):
            try: return data.decode("utf-8")
            except: return data.decode("utf-8", errors="ignore")
        return str(data)
    except Exception:
        return ""

def read_pdf_text(file_like) -> str:
    # 优先 pypdf;失败再试 pdfminer.six
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_like)
        txts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            t = _normalize(t)
            if t: txts.append(t)
        return "\n".join(txts)
    except Exception:
        try:
            from pdfminer.high_level import extract_text
            if hasattr(file_like, "read"):
                data = file_like.read()
            else:
                data = file_like
            return _normalize(extract_text(io.BytesIO(data)))
        except Exception:
            return ""
def _lazy_import_vec():
    """
    兼容旧代码的占位函数：
    返回 (np, faiss, SentenceTransformer, FastEmbedModel, TfidfVectorizer, extra)
    实际上你现在系统只用 get_embedder，不再依赖这个函数的输出。
    """
    import numpy as np
    try:
        import faiss
    except Exception:
        faiss = None

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None

    try:
        from fastembed import TextEmbedding as FastEmbedModel  # 如果没有 fastembed 也无所谓
    except Exception:
        FastEmbedModel = None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        TfidfVectorizer = None

    return np, faiss, SentenceTransformer, FastEmbedModel, TfidfVectorizer, None

# ========== 向量召回(多后端:Sentence-Transformers → Fastembed → TF-IDF)==========
@st.cache_resource(show_spinner=False)
def get_embedder():
    """
    返回 (backend, encode):

        - backend: 固定 "st"
        - encode:  encode(texts: list[str]) -> np.ndarray[float32] (L2 归一化)

    只使用 SentenceTransformer 句向量；
    不再退回 TF-IDF，一旦失败直接报错。
    """
    import numpy as np

    # 1) 导入 SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        st.error(f"❌ 无法导入 sentence-transformers，请先安装依赖: {e}")
        # 这里直接抛错，让调用方在日志里看到真实问题
        raise RuntimeError("sentence-transformers not available") from e

    # 2) 固定使用一个模型（你一直在用的那只）
    model_name = "distiluse-base-multilingual-cased-v1"

    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        st.error(f"❌ 加载句向量模型 {model_name} 失败: {e}")
        raise RuntimeError(f"failed to load sentence transformer model {model_name}") from e

    # 3) 封装统一 encode 函数
    def encode_st(texts: list[str]):
        if not texts:
            # 空输入时返回 (0, dim) 避免后面 shape 异常
            dim = model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype="float32")

        emb = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            convert_to_numpy=True,
        ).astype("float32")

        # 双保险再归一化一次
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return (emb / norms).astype("float32")

    st.info(f"✅ 已启用 SentenceTransformer 句向量: {model_name}")
    return "st", encode_st

def _load_index(project_id: int):
    np, faiss, *_ = _lazy_import_vec()
    idx_path, map_path, vec_path = _index_paths(project_id)
    mapping = []
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    # FAISS
    if faiss is not None and os.path.exists(idx_path):
        index = faiss.read_index(idx_path)
        return ("faiss", index, mapping, None)
    # 回退:.npy
    if os.path.exists(vec_path):
        vecs = np.load(vec_path).astype("float32")
        return ("fallback", None, mapping, vecs)

    # 索引完全不存在
    log_event(
        "WARNING",
        "semantic index not found",
        project_id=project_id,
        idx_path=idx_path,
        vec_path=vec_path,
    )
    return ("none", None, mapping, None)


def _save_index(project_id: int, mode: str, index, mapping, vecs=None):
    np, faiss, *_ = _lazy_import_vec()
    idx_path, map_path, vec_path = _index_paths(project_id)
    if mode == "faiss" and index is not None:
        faiss.write_index(index, idx_path)
    elif mode == "fallback" and vecs is not None:
        np.save(vec_path, vecs.astype("float32"))
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

def _index_paths_domain(domain: str, kb_type: str):
    """按“领域 + 类型”返回对应索引文件路径"""
    domain_key = _norm_domain_key(domain)
    base_dir = os.path.join(BASE_DIR, "semantic_index", domain_key, kb_type)
    os.makedirs(base_dir, exist_ok=True)
    idx_path = os.path.join(base_dir, "index.faiss")
    map_path = os.path.join(base_dir, "mapping.json")
    vec_path = os.path.join(base_dir, "vectors.npy")
    return idx_path, map_path, vec_path


def _load_index_domain(domain: str, kb_type: str):
    """按“领域 + 类型”加载索引. 返回 (mode, index, mapping, vecs)"""
    np, faiss, *_ = _lazy_import_vec()
    idx_path, map_path, vec_path = _index_paths_domain(domain, kb_type)
    mapping = []
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
    if faiss is not None and os.path.exists(idx_path):
        index = faiss.read_index(idx_path)
        return "faiss", index, mapping, None
    if os.path.exists(vec_path):
        vecs = np.load(vec_path).astype("float32")
        return "fallback", None, mapping, vecs
    return "none", None, mapping, None


def _save_index_domain(domain: str, kb_type: str, mode: str, index, mapping, vecs=None):
    np, faiss, *_ = _lazy_import_vec()
    idx_path, map_path, vec_path = _index_paths_domain(domain, kb_type)
    if mode == "faiss" and index is not None:
        faiss.write_index(index, idx_path)
        # 有旧的 numpy 索引就顺手删一下
        if os.path.exists(vec_path):
            try:
                os.remove(vec_path)
            except OSError:
                pass
    elif mode == "fallback" and vecs is not None:
        np.save(vec_path, vecs.astype("float32"))
        # 有旧的 faiss 索引就顺手删一下
        if os.path.exists(idx_path):
            try:
                os.remove(idx_path)
            except OSError:
                pass
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
def build_strategy_index_for_domain(domain: str):
    """为指定领域重建【翻译策略】(strategy) 单语索引.

    数据来源: strategy_texts 表, 每条记录视为一个“策略文段”。
    索引单位: 以整段 content 为一条向量(如需更细粒度, 可以后续再按句子拆分).
    """
    import numpy as _np
    np, faiss, *_ = _lazy_import_vec()
    backend, encode = get_embedder()

    dom = (domain or "").strip() or "未分类"

    # 1) 确保策略表存在
    cur.execute(
        "CREATE TABLE IF NOT EXISTS strategy_texts ("
        "id INTEGER PRIMARY KEY,"
        "domain TEXT,"
        "title TEXT,"
        "content TEXT NOT NULL,"
        "collection TEXT,"
        "source TEXT,"
        "created_at TEXT DEFAULT (datetime('now'))"
        ");"
    )
    conn.commit()

    # 2) 拉取该领域下的全部策略文本
    rows = cur.execute(
        """
        SELECT id,
               IFNULL(domain,''), IFNULL(title,''), content,
               IFNULL(collection,''), IFNULL(source,'')
          FROM strategy_texts
         WHERE IFNULL(domain,'') = ?
         ORDER BY id ASC
        """,
        (dom,)
    ).fetchall()

    texts, metas = [], []
    for sid, d, ttl, content, coll, src in rows:
        txt = (content or "").strip()
        if not txt:
            continue
        texts.append(txt)
        metas.append({
            "strategy_id": sid,
            "domain": d or dom,
            "title": ttl,
            "content_preview": txt[:200],
            "collection": coll,
            "source": src,
            "kb_type": "strategy",
        })

    if not texts:
        # 清空该领域下的策略索引
        _save_index_domain(dom, "strategy", "none", None, [])
        return {"added": 0, "total": 0}

    # 3) 编码向量
    new_vecs = encode(texts)
    if hasattr(new_vecs, "toarray"):
        new_vecs = new_vecs.toarray()
    new_vecs = _np.asarray(new_vecs, dtype="float32")
    if new_vecs.ndim == 1:
        new_vecs = new_vecs.reshape(1, -1)
    new_vecs = new_vecs / (_np.linalg.norm(new_vecs, axis=1, keepdims=True) + 1e-12)

    # 4) 写入索引文件
    if faiss is not None and backend in ("st", "fastembed"):
        dim = int(new_vecs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(new_vecs)
        _save_index_domain(dom, "strategy", "faiss", index, metas)
        total = int(index.ntotal)
    else:
        vecs = new_vecs
        _save_index_domain(dom, "strategy", "fallback", None, metas, vecs=vecs)
        total = int(vecs.shape[0])

    return {"added": len(texts), "total": total}

def build_project_vector_index(project_id: int,
                               use_src: bool = True,
                               use_tgt: bool = True):
    """
    为指定项目所属【领域】重建向量索引(句对版，中英对照):

    - 通过 project_id 找到该项目的 domain；
    - 从 corpus 表中读取该领域下所有双语语料(不再按 project_id 限制)；
    - 按句对对齐: split_sents(src) / split_sents(tgt)；
    - 用“中文句子”作为检索向量文本；
    - mapping 中保存: corpus_id, idx, src, tgt, project_id, domain, title, lang_pair, kb_type;
    - 每次重建时，会覆盖该领域下的双语索引文件(semantic_index/{domain}/bilingual/)。

    返回: {"added": 新增条数, "total": 索引总条数}
    """
    import numpy as _np
    np, faiss, *_ = _lazy_import_vec()
    backend, encode = get_embedder()

    pid = int(project_id)

    # 0) 根据项目取领域
    proj_domain = None
    try:
        row = cur.execute(
            "SELECT IFNULL(domain,'') FROM items WHERE id=?",
            (pid,)
        ).fetchone()
        if row:
            proj_domain = (row[0] or "").strip()
    except Exception:
        proj_domain = None

    if not proj_domain:
        # 没有设置领域时，归入“未分类”
        proj_domain = "未分类"

    # 1) 从 DB 读取该领域下的语料(不再按 project_id 限制)
    rows = cur.execute(
        """
        SELECT c.id,
               IFNULL(c.src_text, ''), IFNULL(c.tgt_text, ''),
               IFNULL(c.title, ''),    IFNULL(c.lang_pair, ''),
               IFNULL(c.project_id, 0), IFNULL(c.domain, '')
        FROM corpus c
        WHERE IFNULL(c.domain, '') = ?
        ORDER BY c.id ASC
        """,
        (proj_domain,)
    ).fetchall()

    texts, metas = [], []

    for cid, s, t, ttl, lp, pj, dom in rows:
        s = (s or "").strip()
        t = (t or "").strip()
        if not s and not t:
            continue

        # 句子切分(尽量使用你已有的 split_sents)
        try:
            if "split_sents" in globals():
                src_sents = split_sents(s, lang_hint="zh")
                tgt_sents = split_sents(t, lang_hint="en")
            else:
                src_sents = (s.split("。") if s else [])
                tgt_sents = (t.split(".") if t else [])
        except Exception:
            src_sents = (s.split("。") if s else [])
            tgt_sents = (t.split(".") if t else [])

        # 如果要求双向对齐，则取最小长度；否则只看 src
        n = min(len(src_sents), len(tgt_sents)) if (use_src and use_tgt) else len(src_sents or [])

        for idx in range(n):
            src_j = (src_sents[idx] if idx < len(src_sents) else "").strip()
            tgt_j = (tgt_sents[idx] if idx < len(tgt_sents) else "").strip()
            if not src_j:
                continue

            texts.append(src_j)
            metas.append({
                "corpus_id": cid,
                "idx": idx,
                "src": src_j,
                "tgt": tgt_j,
                "project_id": pj,
                "domain": dom or proj_domain or "",
                "title": ttl,
                "lang_pair": lp or "",
                "kb_type": "bilingual",
            })

    if not texts:
        # 该领域没有可用语料；清理索引文件，避免残留旧索引
        try:
            _save_index(pid, "none", None, [], vecs=None)
        except Exception:
            pass
        return {"added": 0, "total": 0}

    # 2) 编码 & 归一化
    new_vecs = encode(texts)
    if hasattr(new_vecs, "toarray"):
        new_vecs = new_vecs.toarray()
    new_vecs = _np.asarray(new_vecs, dtype="float32")
    if new_vecs.ndim == 1:
        new_vecs = new_vecs.reshape(1, -1)
    new_vecs = new_vecs / (_np.linalg.norm(new_vecs, axis=1, keepdims=True) + 1e-12)

    # 3) 直接重建索引(不再读取旧 mapping)
    if faiss is not None and backend in ("st", "fastembed"):
        dim = int(new_vecs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(new_vecs)
        _save_index(pid, "faiss", index, metas)
        total = int(index.ntotal)
    else:
        vecs = new_vecs
        _save_index(pid, "fallback", None, metas, vecs=vecs)
        total = int(vecs.shape[0])

    return {"added": len(texts), "total": total}

def rebuild_project_semantic_index(project_id: int) -> dict:
    """
    统一对外的“重建语义索引入口函数”。

    用途：
      - 在 Streamlit UI 之外的脚本里调用；
      - 以后如果需要加日志 / 权限控制，只改这个函数即可。

    参数：
      project_id: 项目 ID（int 或可以转成 int 的字符串）

    返回：
      {"ok": True/False, "added": int, "total": int, "msg": str}
    """
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"ok": False, "added": 0, "total": 0, "msg": f"非法项目ID: {project_id!r}"}

    try:
        res = build_project_vector_index(pid, use_src=True, use_tgt=True)
        return {
            "ok": True,
            "added": int(res.get("added", 0)),
            "total": int(res.get("total", 0)),
            "msg": "索引重建成功",
        }
    except Exception as e:
        return {
            "ok": False,
            "added": 0,
            "total": 0,
            "msg": f"索引重建失败: {e}",
        }

# =========================
# 语义召回(支持范围:project/domain/all)
# 返回: [(score, meta, src_sent, tgt_sent)]
# =========================
def _get_domain_for_proj(cur, project_id: int) -> str | None:
    """
    根据项目ID获取项目领域(domain)，用于 scope="domain" 时过滤参考语料。
    """
    try:
        row = cur.execute(
            "SELECT IFNULL(domain,'') FROM items WHERE id=?",
            (int(project_id),),
        ).fetchone()
    except Exception:
        return None

    if not row:
        return None

    dom = (row[0] or "").strip()
    return dom or None

def semantic_retrieve(project_id: int,
                      query_text: str,
                      topk: int = 20,
                      scope: str = "project",
                      min_char: int = 3):
    """
    语料库语义召回(句级，中英句对)

    统一接口约定：
    ----------------------------------------
    入参：
      - project_id : 当前项目 ID
      - query_text : 查询文本（通常是当前段的中文）
      - topk       : 最终返回的最多条数
      - scope      : "project" / "domain" / "all"
      - min_char   : 最小字符数门槛

    返回：
      List[Tuple[float, dict, str, str]]
      即：[(score, meta, src_sent, tgt_sent), ...]
        - score     : 相似度分数(float，已排序，越大越相似)
        - meta      : 来自 mapping 的字典（至少包含 corpus_id, idx, project_id, domain, title, lang_pair 等）
        - src_sent  : 参考语料中的中文句子
        - tgt_sent  : 参考语料中的对应译文句子（如无则可能为空串）

    注意：
      - 外部只需要记住：永远是四元组；想只用前三个就 row[:3]。
    """
    import numpy as np

    q = (query_text or "").strip()
    if len(q) < min_char:
        return []

    # --- 工具：切句（优先用你已有的 split_sents，失败就正则粗切） ---
    def _split(text: str) -> list[str]:
        try:
            if "split_sents" in globals():
                segs = split_sents(text, lang_hint="auto")  # type: ignore
                return [s for s in segs if s and len(s.strip()) >= min_char]
        except Exception:
            pass
        import re
        segs = re.split(r"(?<=[\.\!\?;。！？；])\s*", text)
        return [s.strip() for s in segs if s and len(s.strip()) >= min_char]

    pieces = _split(q)
    if not pieces:
        return []

    # --- 取向量编码器 ---
    try:
        backend, encode = get_embedder()
    except RuntimeError as e:
        st.error(f"向量模型加载失败: {e}")
        return

    # --- 取索引 & mapping ---
    mode, index, mapping, vecs = _load_index(int(project_id))  # 已在你文件里定义
    if mode == "none" or not mapping:
        return []

    # 懒加载 numpy / faiss
    np_mod, faiss, *_ = _lazy_import_vec()  # 你已有的工具函数
    np = np_mod  # 为了少敲几个字

    # 为了 domain scope，用一下项目领域
    cur_domain = None
    if scope == "domain":
        try:
            cur_domain = _get_domain_for_proj(cur, int(project_id))  # 你文件里已有
        except Exception:
            cur_domain = None

    def _scope_ok(meta: dict) -> bool:
        """按 scope 过滤候选。"""
        if scope == "project":
            return int(meta.get("project_id", 0) or 0) == int(project_id)
        elif scope == "domain" and cur_domain:
            return (meta.get("domain") or "") == (cur_domain or "")
        else:
            # "all" 或拿不到 domain 时，都不过滤
            return True

    all_hits: list[tuple[float, dict, str, str]] = []

    # 为了稳一点，我们让每个查询句子多拿一点候选，再整体去重、截断
    per_piece_k = max(topk * 3, topk)

    for piece in pieces:
        if not piece:
            continue

        try:
            # 1) 生成查询向量 qv
            qv = encode([piece])
            if hasattr(qv, "toarray"):  # tf-idf 稀疏矩阵
                qv = qv.toarray()
            qv = np.asarray(qv, dtype="float32")
            if qv.ndim == 2:
                qv = qv[0]

            # 查询向量归一化
            q_norm = np.linalg.norm(qv) + 1e-12
            qv = qv / q_norm

            # 2) FAISS 分支
            if mode == "faiss" and index is not None and faiss is not None:
                k = min(per_piece_k, len(mapping))
                if k <= 0:
                    continue
                D, I = index.search(qv.reshape(1, -1), k)
                for score, idx in zip(D[0].tolist(), I[0].tolist()):
                    idx = int(idx)
                    if idx < 0 or idx >= len(mapping):
                        continue
                    meta = mapping[idx] or {}
                    if not isinstance(meta, dict):
                        continue
                    if not _scope_ok(meta):
                        continue
                    src_sent = (meta.get("src") or "").strip()
                    tgt_sent = (meta.get("tgt") or "").strip()
                    if not src_sent and not tgt_sent:
                        continue
                    all_hits.append((float(score), meta, src_sent, tgt_sent))

            # 3) fallback 分支：纯 numpy 相似度
            elif mode == "fallback" and vecs is not None:
                arr = np.asarray(vecs, dtype="float32")
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                sims = arr @ qv.reshape(-1, 1)  # 内积，向量已归一化 => cos 相似度
                sims = sims.reshape(-1)
                k = min(per_piece_k, sims.shape[0])
                if k <= 0:
                    continue
                idxs = np.argsort(-sims)[:k]
                for idx in idxs:
                    idx = int(idx)
                    score = float(sims[idx])
                    if idx < 0 or idx >= len(mapping):
                        continue
                    meta = mapping[idx] or {}
                    if not isinstance(meta, dict):
                        continue
                    if not _scope_ok(meta):
                        continue
                    src_sent = (meta.get("src") or "").strip()
                    tgt_sent = (meta.get("tgt") or "").strip()
                    if not src_sent and not tgt_sent:
                        continue
                    all_hits.append((score, meta, src_sent, tgt_sent))

            else:
                # 没有可用索引
                continue

        except Exception:
            # 召回失败时，宁可少给结果，也不要把异常直接炸到 UI
            continue

    if not all_hits:
        return []

    # --- 去重 + 按得分排序，保留前 topk ---
    dedup = {}
    for score, meta, src_sent, tgt_sent in all_hits:
        key = (src_sent, tgt_sent)
        if key not in dedup or score > dedup[key][0]:
            dedup[key] = (score, meta, src_sent, tgt_sent)

    hits = sorted(dedup.values(), key=lambda x: x[0], reverse=True)
    return hits[:topk]

def semantic_consistency_report(project_id: int,
                                blocks_src: list,
                                blocks_tgt: list,
                                term_map: dict,
                                topk: int = 3,
                                thr: float = 0.70):
    """
    译后一致性报告 (语义 + 术语，按段落)。

    参数:
        project_id : 项目ID
        blocks_src : 源文分段列表
        blocks_tgt : 译文分段列表
        term_map   : {源术语: 目标术语}
        topk       : 语义参考检索的候选条数
        thr        : 认为“相似度过低”的阈值

    返回:
        pandas.DataFrame，列包括:
            - 段号
            - 相似参考得分
            - 低于阈值 (bool)
            - 未遵守术语 (逗号分隔的字符串)
    """
    hits_all = []

    # 对齐长度，避免两侧长度不一致出错
    n = min(len(blocks_src or []), len(blocks_tgt or []))
    if n == 0:
        return pd.DataFrame([])

    for i, (s, t) in enumerate(zip(blocks_src[:n], blocks_tgt[:n]), 1):
        s = s or ""
        t = t or ""

        # 1) 用“译文”去检索参考译文（更贴近人工审校）
        try:
            hits = semantic_retrieve(project_id, t, topk=topk)
        except Exception:
            hits = []

        # semantic_retrieve 统一返回: (score, meta, src_sent, tgt_sent)
        if hits:
            top_score = float(hits[0][0])
        else:
            top_score = 0.0

        # 2) 术语遵守：源段包含源术语，但译文里没出现目标术语
        violated = []
        for src_term, tgt_term in (term_map or {}).items():
            if not src_term or not tgt_term:
                continue
            if src_term in s and tgt_term not in t:
                violated.append(f"{src_term}->{tgt_term}")

        hits_all.append(
            {
                "段号": i,
                "相似参考得分": round(top_score, 2),
                "低于阈值": (top_score < thr),
                "未遵守术语": ", ".join(violated) if violated else "",
            }
        )

    return pd.DataFrame(hits_all)

# ========== 路径/DB ==========
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()

def ensure_domain_columns_and_backfill(conn, cur, corpus_table="corpus"):
    # items.domain
    cols = [r[1] for r in cur.execute("PRAGMA table_info(items)").fetchall()]
    if "domain" not in cols:
        cur.execute("ALTER TABLE items ADD COLUMN domain TEXT;")
        conn.commit()

    # 语料表 domain(如果你用 corpus_main 就把参数改成 corpus_main)
    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({corpus_table})").fetchall()]
    if "domain" not in cols:
        cur.execute(f"ALTER TABLE {corpus_table} ADD COLUMN domain TEXT;")
        conn.commit()

    # 回填:用 items.domain 补 corpus.domain(有 project_id 的行)
    try:
        cur.execute(f"""
            UPDATE {corpus_table}
            SET domain = (
              SELECT i.domain FROM items i WHERE i.id = {corpus_table}.project_id
            )
            WHERE domain IS NULL AND project_id IS NOT NULL;
        """)
        conn.commit()
    except Exception:
        pass

# 调用(老库表名是 corpus):
ensure_domain_columns_and_backfill(conn, cur, corpus_table="corpus")
# 若你已经切到 corpus_main / corpus_vec:
# ensure_domain_columns_and_backfill(conn, cur, corpus_table="corpus_main")

def _get_domain_for_proj(cur, project_id: int):
    """
    工具函数: 根据项目ID读取 items.domain; 若不存在或为空, 返回 None。
    """
    try:
        row = cur.execute(
            "SELECT domain FROM items WHERE id=?",
            (int(project_id),)
        ).fetchone()
        if not row:
            return None
        val = (row[0] or "").strip()
        return val or None
    except Exception:
        return None
try:
    cur.execute("CREATE INDEX IF NOT EXISTS idx_term_ext_project ON term_ext(project_id)")
    conn.commit()
except Exception as e:
    print("索引创建跳过:", e)

def _has_col(table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def ensure_col(table: str, col: str, col_type: str):
    """
    确保指定表存在某列；如不存在则添加并立即提交。
    依赖全局的 conn/cur，调用方无需单独 commit。
    """
    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        conn.commit()

# —— 建表
cur.execute("""
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT,
    tags TEXT,
    domain TEXT,
    type TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS item_ext (
    id INTEGER PRIMARY KEY,
    item_id INTEGER,
    src_path TEXT,
    FOREIGN KEY(item_id) REFERENCES items(id)
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS project_files (
    id INTEGER PRIMARY KEY,
    project_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    file_name TEXT,
    uploaded_at TEXT DEFAULT (datetime('now')),
    note TEXT,
    FOREIGN KEY(project_id) REFERENCES items(id)
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS term_ext (
    id INTEGER PRIMARY KEY,
    source_term TEXT NOT NULL,
    target_term TEXT,
    domain TEXT,
    project_id INTEGER,
    strategy TEXT,
    example TEXT,
    category TEXT,
    FOREIGN KEY(project_id) REFERENCES items(id)
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS trans_ext (
    id INTEGER PRIMARY KEY,
    project_id INTEGER,
    src_text TEXT,
    tgt_text TEXT,
    mode TEXT,
    segment_count INTEGER,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(project_id) REFERENCES items(id)
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS corpus (
    id INTEGER PRIMARY KEY,
    project_id INTEGER,
    text TEXT,
    lang TEXT,
    source TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY(project_id) REFERENCES items(id)
);
""")
conn.commit()


# —— 兜底补列
for t, cols in {
    "items": [("type","TEXT"),("tags","TEXT"),("scene","TEXT"),("prompt","TEXT"),
              ("mode","TEXT"),("body","TEXT"),("created_at","TEXT"),("updated_at","TEXT"),("trans_type","TEXT")],
    "item_ext": [("src_path","TEXT")],
    "term_ext": [("domain","TEXT"),("project_id","INTEGER"),("strategy","TEXT"),
                 ("example","TEXT"),("note","TEXT"), ("category","TEXT")],
    "trans_ext": [("stats_json","TEXT"),("segments","INTEGER"),("term_hit_total","INTEGER")],
    "corpus": [
        ("title","TEXT"),("project_id","INTEGER"),("lang_pair","TEXT"),("src_text","TEXT"),("tgt_text","TEXT"),
        ("note","TEXT"),("created_at","TEXT"),("domain","TEXT"),("source","TEXT"),
    ],
}.items():
    for c, tp in cols:
        ensure_col(t, c, tp)
cur.execute("UPDATE items SET type='project' WHERE IFNULL(type,'')=''")
cur.execute("UPDATE items SET created_at = COALESCE(created_at, strftime('%Y-%m-%d %H:%M:%S','now'))")
conn.commit()
ensure_col("term_ext", "example_vector_id", "INTEGER")

# --- 术语表字段兼容:缺少 project_id 时补建 ---
try:
    cur.execute("PRAGMA table_info(term_ext)")
    cols = [c[1] for c in cur.fetchall()]
    if "project_id" not in cols:
        cur.execute("ALTER TABLE term_ext ADD COLUMN project_id INTEGER")
        conn.commit()
except Exception as e:
    st.warning(f"术语表结构检查:{e}")

# ========== DeepSeek 参数/调用 ==========
def get_deepseek():
    """
    从 .streamlit/secrets.toml 读取:
    [deepseek]
    api_key="..."
    model="deepseek-chat"
    """
    try:
        ak = st.secrets["deepseek"]["api_key"]
        model = st.secrets["deepseek"].get("model", "deepseek-chat")
        return ak, model
    except Exception:
        return None, None

        # === 新增:术语提示 + 参考例句 ===
def _build_ref_context(project_id: int,
                       query_text: str,
                       topk: int = 20,
                       min_sim: float = 0.25,
                       prefer_side: str = "both",   # 当前暂未使用，保留以兼容旧参数
                       scope: str = "project",
                       top_n: int = 5) -> str:
    """
    构建参考例句(句级，中英对照)。

    依赖 semantic_retrieve 返回: (score, meta, src_sent, tgt_sent)
    返回:
        一段可直接注入 Prompt 的字符串，多条例句用换行拼接。
    """
    try:
        hits = semantic_retrieve(
            project_id,
            query_text,
            topk=topk,
            scope=scope,
        )
    except Exception as e:
        # 召回失败不影响主流程，只在 UI 环境下做个轻提示
        try:
            st.warning(f"参考检索失败: {e}")
        except Exception:
            pass
        return ""

    if not hits:
        return ""

    # 1) 过滤低相似度 + 去重
    seen = set()
    selected = []

    for sc, meta, src_sent, tgt_sent in hits:
        try:
            score = float(sc or 0.0)
        except Exception:
            score = 0.0

        if score < min_sim:
            continue

        ch = (src_sent or "").strip()
        en = (tgt_sent or "").strip()

        # 中英文都空，就跳过
        if not ch and not en:
            continue

        key = (ch, en)
        if key in seen:
            continue
        seen.add(key)

        selected.append((score, meta, ch, en))
        if len(selected) >= top_n:
            break

    # 如果筛完一个都没有，就退一步：拿最高分那条
    if not selected:
        best = hits[0]
        sc, meta, ch, en = best
        ch = (ch or "").strip()
        en = (en or "").strip()
        if not ch and not en:
            return ""
        try:
            sc = float(sc or 0.0)
        except Exception:
            sc = 0.0
        selected = [(sc, meta, ch, en)]

    # 2) 拼成多行文本
    ctx_lines = ["参考例句(用于保持术语与风格一致):"]
    for idx, (sc, meta, ch, en) in enumerate(selected, 1):
        dom = (meta.get("domain") or "").strip() if isinstance(meta, dict) else ""
        title = (meta.get("title") or "").strip() if isinstance(meta, dict) else ""
        tag_info = " · ".join(x for x in [dom, title] if x)

        ch_show = ch.replace("\n", " ").strip()
        en_show = en.replace("\n", " ").strip()

        if en_show:
            line = (
                f"例句{idx} 原文:{ch_show}\n"
                f"       译文:{en_show}"
                f"(sim={sc:.2f}{'，'+tag_info if tag_info else ''})"
            )
        else:
            line = f"例句{idx}:{ch_show}(sim={sc:.2f}{'，'+tag_info if tag_info else ''})"

        ctx_lines.append(line)

        # 控制总长度，避免 prompt 过长
        if sum(len(x) for x in ctx_lines) > 1800:
            break

    return "\n".join(ctx_lines) if len(ctx_lines) > 1 else ""
# -------- Glossary & Instruction helpers (放在 ds_translate 上方) --------
def build_term_hint(term_dict: dict, lang_pair: str, max_terms: int = 80) -> str:
    """
    将术语映射转成可读的“硬约束”规则文本.支持以下几种 term_dict 结构:
      { "contract": "合同" }
      { "contract": {"target":"合同", "pos":"NOUN", "usage_note":"法律语境"} }
      { "contract": ("合同", "NOUN") }   # 元组形式 (target, pos)
    空/非 dict 的输入会被安全忽略; 空目标会被忽略; 自动去重并最多输出
    max_terms 条，避免提示过长。
    """
    if not term_dict or not isinstance(term_dict, dict):
        return ""

    lines = []
    seen = set()
    items = list(term_dict.items())[: max_terms * 2]  # 稍多取一些.过滤空后再截断

    for src, val in items:
        if src is None: 
            continue
        src = str(src).strip()
        if not src or src in seen:
            continue

        tgt, pos, note = None, None, None
        if isinstance(val, dict):
            tgt  = (val.get("target") or val.get("tgt") or "").strip()
            pos  = (val.get("pos") or "").strip() or None
            note = (val.get("usage_note") or val.get("note") or "").strip() or None
        elif isinstance(val, (list, tuple)) and len(val) >= 1:
            tgt = str(val[0]).strip()
            if len(val) >= 2:
                pos = (str(val[1]).strip() or None)
        else:
            tgt = str(val or "").strip()

        if not tgt:
            continue

        seen.add(src)
        if pos:
            line = f"- When '{src}' is a {pos}, translate it as '{tgt}'."
        else:
            line = f"- Translate '{src}' as '{tgt}'."
        if note:
            line += f" ({note})"
        lines.append(line)

    if not lines:
        return ""  # 没有可用术语就返回空串.不要干扰提示

    header = "GLOSSARY (STRICT):\n"
    return header + "\n".join(lines[:max_terms]) + "\n"


def build_instruction(lang_pair: str) -> str:
    """
    生成简洁的翻译指令。你也可以按项目风格再扩展。

    - 支持中文与英文写法（如 "Chinese to English"/"English→Chinese"）。
    - 统一把各种箭头/连字符/"to" 转成 "-"，便于模式匹配。
    """
    lp_raw = (lang_pair or "").replace(" ", "")
    lp_norm = lp_raw.lower()
    for sep in ("→", "->", "=>", "—>", "—", "—", "—-", "——"):
        lp_norm = lp_norm.replace(sep, "-")
    lp_norm = (
        lp_norm.replace("to", "-")
        .replace("_", "-")
        .replace("/", "-")
    )

    zh_to_en_tokens = (
        "中译英", "中→英", "中->英", "中-英", "zh-en", "zh2en", "zh_en", "zh-en",
        "chinese-english", "chinese-en", "zh-english",
    )
    en_to_zh_tokens = (
        "英译中", "英→中", "英->中", "英-中", "en-zh", "en2zh", "en_zh", "en-zh",
        "english-chinese", "english-zh", "en-chinese",
    )

    def _match(tokens: tuple[str, ...]) -> bool:
        return any(tok in lp_raw or tok in lp_norm for tok in tokens)

    if _match(zh_to_en_tokens):
        return (
            "Translate the source text from Chinese to English. "
            "Use a professional, natural style; follow the GLOSSARY (STRICT) exactly; "
            "preserve proper nouns and numbers; keep paragraph structure. "
            "Do not add explanations."
        )

    if _match(en_to_zh_tokens):
        return (
            "Translate the source text from English to Chinese. "
            "用专业、通顺、符合领域文体的中文表达;严格遵守上方 GLOSSARY (STRICT);"
            "专有名词、数字与计量单位保持准确;段落结构保持一致。不得添加解释。"
        )

    # 兜底
    return (
        "Translate the source text. Follow the GLOSSARY (STRICT) exactly. "
        "Keep the original structure and do not add explanations."
    )

def ds_translate(
    block: str,
    term_dict: dict,
    lang_pair: str,
    ak: str,
    model: str,
    ref_context: str = "",
    fewshot_examples=None,
) -> str:
    term_hint = build_term_hint(term_dict, lang_pair)  # 统一使用严格术语提示
    instr = build_instruction(lang_pair)   # type: ignore

    """
    使用 DeepSeek REST API 翻译一个文本块。term_dict 为 {源: 目标} 的映射.注入为强约束提示。
    """
    import requests

    if not block.strip():
        return ""

    # 如果术语为空，为了让提示始终包含 GLOSSARY 段落，给出一个安全的兜底
    if not term_hint:
        if term_dict:
            # 术语字典存在但内容被过滤为空，给出简洁的默认提示
            term_hint = (
                "GLOSSARY (STRICT):\n"
                "- Follow provided terminology exactly; do not paraphrase fixed terms.\n\n"
            )
        else:
            term_hint = (
                "GLOSSARY (STRICT):\n"
                "- Ensure consistent terminology; avoid paraphrasing fixed terms.\n\n"
            )

    # 保证与后续 INSTRUCTION 块之间有空行
    if not term_hint.endswith("\n\n"):
        term_hint = term_hint.rstrip("\n") + "\n\n"

    system_msg = (
        "You are a senior professional translator. Prioritize accuracy, faithfulness, and consistent terminology. "
        "No hallucinations. If a term mapping is provided, follow it strictly."
    )

    user_msg = (
        f"{term_hint}"
        + (f"REFERENCE CONTEXT (use if relevant):\n{ref_context}\n\n" if ref_context else "")
        + "INSTRUCTION:\n" + instr + "\n"
        + "RESPONSE FORMAT:\n- Return ONLY the final translation text, no explanations, no backticks.\n\n"
        + "SOURCE:\n" + block
    )

    messages = [{"role": "system", "content": system_msg}]
    if fewshot_examples:
        for ex in fewshot_examples:
            src_demo = (ex.get("src") or "").strip()
            tgt_demo = (ex.get("tgt") or "").strip()
            if not (src_demo and tgt_demo):
                continue
            title = ex.get("title") or ""
            demo_user = f"【参考示例:{title}】\n源文:\n{src_demo}"
            messages.append({"role": "user", "content": demo_user})
            messages.append({"role": "assistant", "content": tgt_demo})
    messages.append({"role": "user", "content": user_msg})

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {ak}", "Content-Type": "application/json"}
    payload = {
        "model": model or "deepseek-chat",
        "messages": messages,
        "temperature": 0.2,
    }

    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                txt = f"[DeepSeek {resp.status_code}] {resp.text}"
                if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                # 最终失败: 记一条错误日志
                log_event(
                    "ERROR",
                    "DeepSeek HTTP error",
                    status_code=resp.status_code,
                    body=resp.text[:500],
                )
                return txt
        except Exception as e:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            # 重试用尽仍然异常: 记一条错误日志
            log_event(
                "ERROR",
                "DeepSeek request exception",
                error=str(e),
            )
            return f"[DeepSeek Request Error] {e}"

    # 理论上不会走到这一步，如果走到了，也记一条
    log_event("ERROR", "DeepSeek unknown failure")
    return "[DeepSeek Error] Unknown failure."


def translate_block_with_kb(
    cur,
    project_id: int,
    block_text: str,
    lang_pair: str,
    ak: str,
    model: str,
    use_semantic: bool = True,
    scope: str = "project",
    fewshot_examples=None,
):
    """
    单段翻译主管线(核心接口之一)。

    功能:
        block_text -> (术语 + 参考例句) -> DeepSeek 翻译 -> 结构化结果

    参数:
        cur            : SQLite cursor
        project_id     : 当前项目 ID
        block_text     : 源段落文本
        lang_pair      : "中译英" / "英译中"
        ak, model      : DeepSeek 的 key 和模型名
        use_semantic   : 是否启用语义召回参考
        scope          : 语义召回范围("project"/"domain"/"all")
        fewshot_examples: few-shot 示例(同 ds_translate)

    返回:
        result: dict，字段包括:
            - src               : 源文本
            - tgt               : 译文
            - project_id        : 项目ID
            - lang_pair         : 翻译方向
            - term_map_all      : 全量静态术语 {源: 目标}
            - terms_in_block    : 当前段落文本中命中的术语 {源: 目标}
            - terms_corpus_dyn  : 参考例句中命中的术语 {源: 目标}
            - terms_final       : 最终注入 Prompt 的术语 {源: 目标}
            - term_meta         : 术语元信息列表(来源/领域等)
            - ref_context       : 注入的参考例句文本(如果启用语义召回)
            - violated_terms    : 粗略一致性检查中“可能未遵守”的术语列表
    """
    blk = (block_text or "").strip()
    if not blk:
        return {
            "src": "",
            "tgt": "",
            "project_id": project_id,
            "lang_pair": lang_pair,
            "term_map_all": {},
            "terms_in_block": {},
            "terms_corpus_dyn": {},
            "terms_final": {},
            "term_meta": [],
            "ref_context": "",
            "violated_terms": [],
        }

    # 1) 静态术语（项目+全局），统一接口
    term_map_all, term_meta = get_terms_for_project(cur, project_id, use_dynamic=True)

    # 2) 命中检测工具：给“本段”和“语料参考”共用
    def _detect_hits(text: str, term_map: dict[str, str]) -> dict[str, str]:
        txt_low = (text or "").lower()
        out = {}
        for k, v in (term_map or {}).items():
            if not k:
                continue
            key_low = k.lower()
            if key_low in txt_low or k in text:
                out[k] = v
        return out

    # 2.1 当前段落文本中命中的术语
    terms_in_block = _detect_hits(blk, term_map_all)

    # 3) 参考例句(来自语料库语义召回)
    if use_semantic:
        try:
            ref_context = _build_ref_context(
                project_id,
                blk,
                topk=20,
                min_sim=0.25,
                prefer_side="both",
                scope=scope,
            )
        except Exception:
            # 召回失败不影响主流程
            ref_context = ""
    else:
        ref_context = ""

    # 3.1 语料驱动术语：在参考例句文本中命中的静态术语
    if ref_context:
        terms_corpus_dyn = _detect_hits(ref_context, term_map_all)
    else:
        terms_corpus_dyn = {}

    # 3.2 最终注入 Prompt 的术语 = 两者并集
    terms_final = dict(terms_in_block)
    for k, v in terms_corpus_dyn.items():
        if k not in terms_final:
            terms_final[k] = v

    # 4) 调用 DeepSeek 翻译（只喂最终术语）
    tgt = ds_translate(
        block=blk,
        term_dict=terms_final,
        lang_pair=lang_pair,
        ak=ak,
        model=model,
        ref_context=ref_context,
        fewshot_examples=fewshot_examples,
    )

    # 5) 粗略术语一致性检查(以“最终注入”的术语为准)
    violated = check_term_consistency(tgt, terms_final, blk)

    return {
        "src": blk,
        "tgt": tgt,
        "project_id": project_id,
        "lang_pair": lang_pair,
        "term_map_all": term_map_all,
        "terms_in_block": terms_in_block,
        "terms_corpus_dyn": terms_corpus_dyn,
        "terms_final": terms_final,
        "term_meta": term_meta,
        "ref_context": ref_context,
        "violated_terms": violated,
    }

def _split_sentences_for_terms(text: str) -> list[str]:
    """用于术语示例抽取的轻量分句，兼容中英文标点。"""
    if not text:
        return []
    txt = _norm_text(text)
    if not txt:
        return []
    parts = re.split(r"(?<=[。！？；.!?])\s+|\n+", txt)
    return [p.strip() for p in parts if p.strip()]


def _locate_example_pair(example: str | None, src_full: str | None, tgt_full: str | None):
    """
    在翻译历史中为示例句找到可能的对齐译文。
    返回 (src_example, tgt_example or None)。
    """
    if not example:
        return None, None

    ex = example.strip()
    if not ex:
        return None, None

    src_sents = split_sents(src_full or "", prefer_newline=True, min_char=2)
    tgt_sents = split_sents(tgt_full or "", prefer_newline=True, min_char=1)

    match_idx = None
    for i, s in enumerate(src_sents):
        if ex in s:
            match_idx = i
            break

    if match_idx is None:
        return ex, None

    tgt = tgt_sents[match_idx] if match_idx < len(tgt_sents) else None
    return ex, tgt or None


def extract_terms_with_corpus_model(
    text: str,
    *,
    max_terms: int = 30,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    default_domain: str | None = None,
):
    """
    使用与语料库向量检索同一套模型(distiluse-base-multilingual-cased-v1)做术语提取。

    逻辑:
    1) 借助正则从文本中抓取中英术语候选(2-8 字中文、1-3 词英文短语)。
    2) 用 get_embedder() 返回的句向量模型对全文和候选做向量化，按相似度选出代表性术语。
    3) 结构化返回字段与原 DeepSeek 提示保持一致(source/target/domain/strategy/example)。
    """

    txt = (text or "").strip()
    if not txt:
        return []

    backend, encode = get_embedder()

    def _dedup_keep(seq):
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    zh_candidates = re.findall(r"[\u4e00-\u9fa5]{2,8}", txt)
    en_candidates = re.findall(r"[A-Za-z][A-Za-z\-]{2,}(?: [A-Za-z\-]{2,}){0,2}", txt)
    candidates = _dedup_keep(zh_candidates + en_candidates)
    if not candidates:
        return []

    doc_emb = encode([txt])[0]
    cand_emb = encode(candidates)
    scores = cand_emb @ doc_emb

    ranked = sorted(zip(candidates, scores.tolist()), key=lambda x: x[1], reverse=True)[:max_terms]
    sents = _split_sentences_for_terms(txt)

    def _example_for(term: str):
        for s in sents:
            if term in s:
                return s
        return None

    out = []
    domain_val = (default_domain or "").strip() or "其他"

    for term, sc in ranked:
        out.append(
            {
                "source_term": term,
                # 现阶段缺少统一的自动译法，保持字段齐全以便后续人工/模型补全
                "target_term": None,
                "domain": domain_val,
                "strategy": None,
                "example": _example_for(term),
                "score": float(sc),
                "model": backend,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
            }
        )
    return out


def ds_extract_terms(
    text: str,
    ak: str,
    model: str,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    *,
    prefer_corpus_model: bool = True,
    default_domain: str | None = None,
):
    """术语提取：优先走语料库同款向量模型，失败时再回退 DeepSeek Prompt。"""

    txt = (text or "").strip()
    if not txt:
        return []

    if prefer_corpus_model:
        try:
            return extract_terms_with_corpus_model(
                txt,
                max_terms=30,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                default_domain=default_domain,
            )
        except Exception as e:
            log_event("ERROR", "corpus-model term extraction failed", error=str(e))

    if not ak:
        return []

    import requests

    system_msg = (
        "You are a terminology mining assistant. Extract high-value bilingual term pairs suitable for a project glossary. "
        "Return JSON array only. No extra text."
    )
    user_msg = f"""
Source language: {src_lang}
Target language: {tgt_lang}
任务:从给定文本中抽取双语术语条目.输出 JSON 数组。字段名与取值必须是中文。
字段定义:
- source_term: 源语(中文术语或专名)
- target_term: 译文(英文)
- domain: 领域.取值集合之一:["政治","经济","文化","文物","金融","法律","其他"]
- strategy: 翻译策略.取值集合之一:["直译","意译","转译","音译","省略","增译","规范化","其他"]
- example: 例句(原文中包含该术语的一句.尽量保留标点)

要求:
1) 仅输出 JSON.不要多余说明。
2) 同一术语重复时合并.选择最典型的例句。
3) 若无法判断 domain/strategy.填“其他”。

Text:
{text}
"""

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {ak}", "Content-Type": "application/json"}
    payload = {
        "model": model or "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.1,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        txt = data["choices"][0]["message"]["content"].strip()
        # 只保留 JSON 片段
        start = txt.find("[")
        end = txt.rfind("]")
        if start == -1 or end == -1:
            return []
        arr = json.loads(txt[start:end+1])
        out = []
        for o in arr:
            src = (o.get("source_term") or o.get("source") or "").strip()
            tgt = (o.get("target_term") or o.get("target") or "").strip()
            dom = (o.get("domain") or "").strip() or (default_domain or None)
            strat = (o.get("strategy") or "").strip() or None
            ex = (o.get("example") or "").strip() or None
            if src:
                out.append({"source_term": src, "target_term": tgt, "domain": dom, "strategy": strat, "example": ex})
        return out
    except Exception:
        return []

# ========== 文件读写与导出 ==========
def read_source_file(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except Exception:
                continue
        with open(path, "r", errors="ignore") as f:
            return f.read()
    elif ext == ".docx":
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return ""
    elif ext == ".xlsx":
        try:
            xls = pd.ExcelFile(path)
            parts = []
            for sheet in xls.sheet_names:
                df = xls.parse(sheet)
                parts.append(df.astype(str).to_csv(sep=" ", index=False, header=False))
            return "\n".join(parts)
        except Exception:
            return ""
    else:
        # 兜底尝试文本读取
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

def build_bilingual_lines(src_text: str, tgt_text: str):
    """
    用段落做对齐:
    - 每一段中文对应一段英文
    - 段内不再拆句(避免 CSV / Word 错位)
    """
    return pair_paragraphs(src_text, tgt_text)

def export_csv_bilingual(src_text: str, tgt_text: str) -> bytes:
    s, t = build_bilingual_lines(src_text, tgt_text)
    df = pd.DataFrame({"Source": s, "Target": t})
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")

def export_docx_bilingual(src_text: str, tgt_text: str) -> bytes:
    try:
        from docx import Document
        from docx.oxml.ns import qn
    except Exception:
        st.error("缺少 python-docx.请先安装:pip install python-docx")
        return b""
    doc = Document()
    # 基础字体
    try:
        doc.styles['Normal'].font.name = 'Calibri'
        doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), '微软雅黑')
    except Exception:
        pass
    s, t = build_bilingual_lines(src_text, tgt_text)
    table = doc.add_table(rows=1, cols=2)
    hdr = table.rows[0].cells
    hdr[0].text = "Source"
    hdr[1].text = "Target"
    for a, b in zip(s, t):
        row = table.add_row().cells
        row[0].text = a
        row[1].text = b
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()

# ==== 工具:文件读取 / 分句 / 向量 / 对齐 / 索引 ====
import os, re, io, json, numpy as _np

def _lazy_import_doc_pdf():
    docx = pdfplumber = None
    try:
        import docx as _docx
        docx = _docx
    except Exception:
        pass
    try:
        import pdfplumber as _pdfplumber
        pdfplumber = _pdfplumber
    except Exception:
        pass
    return docx, pdfplumber
# 段落切分
def split_paragraphs(text: str) -> list[str]:
    """
    段落切分(用于翻译 & 导出):
    - 统一换行符
    - 以【至少一个空行】作为段落分隔
    - 段内保留句子，只去掉纯空行
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # 常见情况:用“一行一段”的稿子，实际上中间没有空行
    # 这种就按单行当作段落
    if "\n\n" not in text and "\n \n" not in text:
        lines = [ln.strip() for ln in text.split("\n")]
        return [ln for ln in lines if ln]

    # 正常:有空行分段
    parts = re.split(r"\n\s*\n+", text)
    paras = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 段内如果还有软换行，压成空格，避免导出时被拆成多行
        p = re.sub(r"\s*\n\s*", " ", p)
        paras.append(p)
    return paras

def pair_paragraphs(src_full: str, tgt_full: str) -> tuple[list[str], list[str]]:
    """
    根据全文中英，按“段落”配对:
    - 源文/译文各自做 split_paragraphs
    - 行数不一致时用空串补齐
    - 保证导出时一行中文对应一行英文
    """
    src_paras = split_paragraphs(src_full or "")
    tgt_paras = split_paragraphs(tgt_full or "")

    n = max(len(src_paras), len(tgt_paras))
    src_paras += [""] * (n - len(src_paras))
    tgt_paras += [""] * (n - len(tgt_paras))
    return src_paras, tgt_paras

# 预编译(可放全局)
_RE_WS = re.compile(r"[ \t\u00A0\u200B\u200C\u200D]+")
_RE_ZH_SENT = re.compile(r"(?<=[。！？；])\s*")           # 中文句末
_RE_EN_SENT = re.compile(r"(?<=[\.\?\!;:])\s+")          # 英文句末(放宽，不强制大写)
_RE_BLANK_PARA = re.compile(r"\n{2,}")                   # 空行分段

def _norm_text(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x0b", "\n")
    t = _RE_WS.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)  # 过多空行压到两个
    return t.strip()

def _is_zh(text: str) -> bool:
    # 简单判定:含有较多中文字符
    zh_hits = len(re.findall(r"[\u4e00-\u9fff]", text))
    en_hits = len(re.findall(r"[A-Za-z]", text))
    return zh_hits >= en_hits

def split_sents(
    text: str,
    lang_hint: str = "auto",
    min_char: int = 1,
    prefer_newline: bool = True,
    **kwargs,
):
    """
    统一的分句/分段函数:
    - 兼容旧调用:split_sents(text, lang="zh")
    - 支持新参数:prefer_newline=True 时，优先按换行切
    """
    # 兼容旧参数名 lang=
    lang = kwargs.get("lang", lang_hint)

    t = _norm_text(text)
    if not t:
        return []

    pieces = []

    # A) 若文本中有换行 & prefer_newline=True:先按行切，再在行内按句末细分
    if prefer_newline and ("\n" in t):
        lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
        for ln in lines:
            if lang == "auto":
                cur_lang = "zh" if _is_zh(ln) else "en"
            else:
                cur_lang = lang

            if cur_lang.startswith(("zh", "cn")):
                sents = [s.strip() for s in _RE_ZH_SENT.split(ln) if s and s.strip()]
            else:
                sents = [s.strip() for s in _RE_EN_SENT.split(ln) if s and s.strip()]

            pieces.extend(sents if sents else [ln])
    else:
        # B) 没有换行或不偏好换行:整块按句末标点切
        if lang == "auto":
            cur_lang = "zh" if _is_zh(t) else "en"
        else:
            cur_lang = lang

        if cur_lang.startswith(("zh", "cn")):
            pieces = [s.strip() for s in _RE_ZH_SENT.split(t) if s and s.strip()]
        else:
            pieces = [s.strip() for s in _RE_EN_SENT.split(t) if s and s.strip()]

        if not pieces:
            pieces = [t]

    # 过滤过短片段
    return [x for x in pieces if len(x) >= min_char]

# 兼容旧函数名
split_sentences = split_sents

def split_blocks(text: str, max_len: int = 1200):
    blocks, curbuf = [], ""
    for line in text.splitlines(True):
        if len(curbuf) + len(line) > max_len:
            if curbuf:
                blocks.append(curbuf)
            curbuf = ""
        curbuf += line
    if curbuf:
        blocks.append(curbuf)
    return blocks
# =========================
# 术语提示 & 译后一致性检查
# =========================
def check_term_consistency(out_text: str, term_dict: dict, source_text: str = "") -> list:
    """
    译后的一致性粗检:如果源文包含术语键.但译文未出现对应值.则记录提醒。
    仅做最小代价的字符串级检查(不改动原译文)。
    返回形如 ["contract→合同", ...] 的列表;为空表示全部符合/不适用。
    """
    if not out_text or not term_dict:
        return []
    warns = []
    s = (source_text or "")[:2000]  # 限长度.防极端长文本
    out_low = out_text.lower()
    for k, v in term_dict.items():
        if not k or not v:
            continue
        # 如果源文包含该术语键(大小写忽略英文;中文直接包含)
        hit_src = (k.lower() in s.lower()) if any(ord(ch) < 128 for ch in k) else (k in s)
        if hit_src:
            # 译文是否出现目标术语(同理大小写容忍英文)
            ok = (v.lower() in out_low) if any(ord(ch) < 128 for ch in v) else (v in out_text)
            if not ok:
                warns.append(f"{k}→{v}")
    return warns

def _lazy_embedder():
    # 优先 sentence-transformers;失败退化到 TF-IDF
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        mdl = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        def _emb(texts):
            arr = mdl.encode(texts, normalize_embeddings=True)
            return arr.astype("float32")
        return _emb, "sbert"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        def _emb(texts):
            vec = TfidfVectorizer(min_df=1).fit_transform(texts)
            # 归一化
            norms = _np.sqrt((vec.multiply(vec)).sum(axis=1)).A.ravel() + 1e-8
            vec = vec.multiply(1/norms[:,None])
            return vec
        return _emb, "tfidf"
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
    """
    统一的“上传语料 → 写入 corpus → 可选重建索引”管线。

    参数：
    - pid: 关联项目 ID，可为空
    - title: 用户在界面填的标题
    - lp: 方向，如 "中译英" / "英译中" / "自动"
    - pairs: 已经对齐好的 [(src, tgt)] 或 [(src, tgt, score)]
    - src_text, tgt_text: 只有单语时用 src_text，多语对照已经合成 pairs 时可以为 None
    - default_title: 若 title 为空时用的兜底标题（文件名）
    - build_after_import: 是否在写入后重建该项目的语义索引
    """
    # 统一标题
    base_title = (title or default_title or "").strip() or "未命名语料"

    # 小工具：把 [(s,t,score)] / [(s,t)] 统一成 [(s,t)]
    def normalize_pairs_to2(pairs_in):
        if not pairs_in:
            return []
        if len(pairs_in[0]) == 3:
            return [(s, t) for (s, t, _) in pairs_in]
        return pairs_in

    # 1) 双语 pairs 情况
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

        # 可选：导入后重建当前项目语义索引
        if build_after_import and pid:
            res_idx = rebuild_project_semantic_index(pid)
            if res_idx.get("ok"):
                st.success(
                    f"🧠 向量索引已更新: 新增 {res_idx['added']}，总量 {res_idx['total']}。"
                )
            else:
                st.warning(f"索引未更新: {res_idx.get('msg','未知错误')}")

        return

    # 2) 单语 src_text 情况（策略/单语语料）
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
                    "mono",  # 标记：单语策略/语料
                ),
            )
            ins += 1
        conn.commit()
        st.success(f"✅ 已写入语料库 {ins} 条。")

        if build_after_import and pid:
            res_idx = rebuild_project_semantic_index(pid)
            if res_idx.get("ok"):
                st.success(
                    f"🧠 向量索引已更新: 新增 {res_idx['added']}，总量 {res_idx['total']}。"
                )
            else:
                st.warning(f"索引未更新: {res_idx.get('msg','未知错误')}")
        return

    # 3) 其他情况：啥都没有，给出提醒
    st.warning("原文和译文都为空，无法写入语料库。")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def align_semantic(src_sents, tgt_sents, max_jump=3):
    """简单贪心 + 滑窗的 1-1 句对齐.返回 [(src, tgt, score)]"""
    if not src_sents or not tgt_sents:
        return []

    emb, kind = _lazy_embedder()

    # === 优先使用 SBERT ===
    if kind == "sbert":
        E1 = emb(src_sents)
        E2 = emb(tgt_sents)
        sims = E1 @ E2.T  # (n, m)
    else:
        # === TF-IDF 回退:确保同一词表维度 ===
        vec = TfidfVectorizer(
            analyzer="char_wb",  # 字符 n-gram 对中英混合最稳
            ngram_range=(1, 2),
            min_df=1
        )
        combo = src_sents + tgt_sents
        X = vec.fit_transform(combo)
        n = len(src_sents)
        E1 = X[:n, :]
        E2 = X[n:, :]

        # 稀疏→相似度矩阵
        sims = cosine_similarity(E1, E2, dense_output=True)  # shape (n, m)

    # === 贪心对齐 ===
    i = j = 0
    n, m = len(src_sents), len(tgt_sents)
    pairs = []
    while i < n and j < m:
        j_min = max(0, j - max_jump)
        j_max = min(m, j + max_jump + 1)
        window = sims[i, j_min:j_max]
        if window.size == 0:
            break
        k = int(window.argmax())
        j_sel = j_min + k
        score = float(sims[i, j_sel])
        pairs.append((src_sents[i], tgt_sents[j_sel], score))
        i += 1
        j = j_sel + 1
    return pairs

# ========== 术语库管理 ==========
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

def render_index_manager_by_domain(st, conn, cur):
    """领域 + 类型视角的索引管理页面"""
    st.subheader("🧠 领域级索引管理 (双语对照 + 翻译策略)")

    # 1) 收集所有可能的领域值
    domains = set()

    # from items
    try:
        rows = cur.execute("SELECT DISTINCT IFNULL(domain,'未分类') FROM items").fetchall()
        for (d,) in rows:
            if d and d.strip():
                domains.add(d.strip())
            else:
                domains.add("未分类")
    except Exception:
        pass

    # from corpus.domain (如果有该字段)
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(corpus)").fetchall()]
        if "domain" in cols:
            rows = cur.execute("SELECT DISTINCT IFNULL(domain,'未分类') FROM corpus").fetchall()
            for (d,) in rows:
                if d and d.strip():
                    domains.add(d.strip())
                else:
                    domains.add("未分类")
    except Exception:
        pass

    # from strategy_texts(如果已经存在)
    try:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS strategy_texts ("
            "id INTEGER PRIMARY KEY,"
            "domain TEXT,"
            "title TEXT,"
            "content TEXT NOT NULL,"
            "collection TEXT,"
            "source TEXT,"
            "created_at TEXT DEFAULT (datetime('now'))"
            ");"
        )
        conn.commit()
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
        (dom_key,)
    ).fetchall()
    proj_ids = [pid for (pid, _) in proj_rows]

    # 2.2 语料条数(如果 corpus 有 domain 字段)
    corpus_cnt = None
    try:
        cols = [r[1] for r in cur.execute("PRAGMA table_info(corpus)").fetchall()]
        if "domain" in cols:
            corpus_cnt = cur.execute(
                "SELECT COUNT(*) FROM corpus WHERE IFNULL(domain,'未分类') = ?",
                (dom_key,)
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
            (dom_key,)
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
                        res = build_project_vector_index(int(pid))
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
                (dom_key,)
            ).fetchall()
            if not rows:
                st.write("暂无策略文本。")
            else:
                for sid, ttl, preview in rows:
                    st.write(f"**[{sid}] {ttl or '(无标题)'}**")
                    st.write(preview + ("..." if len(preview) >= 200 else ""))
                    st.markdown("---")
        except Exception as e:
            st.write(f"读取策略文本出错: {e}")

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
            proj_domain = _project_domain(pid)

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
                            res_idx = rebuild_project_semantic_index(pid)
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

    _ensure_project_ref_map()
    _ensure_project_switch_map()
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
                res_idx = rebuild_project_semantic_index(pid)
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
                    res_idx = rebuild_project_semantic_index(pid_sel)
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
                hits = semantic_retrieve(pid_sel, q_demo.strip(), topk=int(topk))
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

