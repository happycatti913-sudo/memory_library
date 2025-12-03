# -*- coding: utf-8 -*-
"""项目文件管理与少样本引用工具。

- 上传/清理项目附件
- 记录少样本引用 ID 与开关状态
- 读取少样本语料示例
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

import streamlit as st

from .config import PROJECT_DIR


def register_project_file(cur, conn, project_id, file_name, data_bytes):
    """将上传的文件保存到项目目录，并记录在 ``project_files`` 表中。"""
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
        INSERT INTO project_files (project_id, file_path, file_name, created_at)
        VALUES (?, ?, ?, datetime('now'))
        """,
        (project_id, full_path, safe_name),
    )
    conn.commit()
    return full_path


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
    """将旧版的 ``item_ext.src_path`` 补录到 ``project_files``。"""
    if not legacy_path or not os.path.exists(legacy_path):
        return
    cur.execute(
        """
        INSERT INTO project_files (project_id, file_path, file_name)
        VALUES (?, ?, ?)
        """,
        (project_id, legacy_path, os.path.basename(legacy_path) or None),
    )
    conn.commit()


def _ensure_project_ref_map():
    """确保 ``session_state['corpus_refs']`` 为 ``{project_id: set(ids)}`` 结构。"""
    refs = st.session_state.get("corpus_refs")
    if isinstance(refs, dict):
        return refs
    st.session_state["corpus_refs"] = {}
    return st.session_state["corpus_refs"]


def _ensure_project_switch_map():
    """确保 ``session_state['cor_use_ref']`` 为 ``{project_id: bool}`` 结构。"""
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
