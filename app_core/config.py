# -*- coding: utf-8 -*-
"""基础配置与通用工具。

- 路径与目录常量
- 轻量日志
- 可选的动态术语/向量组件
- 路径归一化与术语去重工具
"""

import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Iterable

import streamlit as st

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

# ======== 轻量日志机制 ========
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "app.log")


def log_event(level: str, message: str, **extra: Any):
    """写入轻量 JSON 日志，失败时静默。"""
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
        pass


# ==== third-party (可选) ====
try:
    from docx import Document  # 在需要处仍会 try/except
except Exception:
    Document = None

# ========== kb_dynamic (可选) ==========
KBEmbedder = None
recommend_for_segment = None
build_prompt_strict = None
build_prompt_soft = None
try:  # pragma: no cover - 动态可选组件
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


# ======= 通用工具 ========
def make_sk(prefix: str):
    """返回一个带有前缀的 key 生成器"""

    return lambda name, id=None: f"{prefix}_{name}_{id}" if id else f"{prefix}_{name}"


# 全局默认 key 生成器(替代被删除的计数器版 sk)
sk = make_sk("global")


def _norm_domain_key(raw: str | None) -> str:
    """把数据库里的 domain 字段转成适合作为文件夹名的 key。"""

    s = (raw or "").strip()
    if not s:
        s = "未分类"
    for ch in r"\\/:\"*?<>|":
        s = s.replace(ch, "_")
    return s


def _project_domain(pid: int | None, cur=None) -> str | None:
    """安全获取项目的领域标签，支持可选 cursor。"""

    if not pid or cur is None:
        return None
    try:
        row = cur.execute("SELECT IFNULL(domain,'') FROM items WHERE id=?", (int(pid),)).fetchone()
        dom = (row[0] if row else "").strip()
        return dom or None
    except Exception:
        return None


def _index_paths(project_id: int, cur=None):
    """统一的语义索引路径(按“领域/类型”归类)。"""

    domain_raw = None
    if cur is not None:
        try:
            row = cur.execute(
                "SELECT IFNULL(domain,'') FROM items WHERE id=?",
                (int(project_id),),
            ).fetchone()
            if row:
                domain_raw = (row[0] or "").strip()
        except Exception:
            domain_raw = None

    domain_key = _norm_domain_key(domain_raw)
    kb_type = "bilingual"

    base_dir = os.path.join(SEM_INDEX_ROOT, domain_key, kb_type)
    os.makedirs(base_dir, exist_ok=True)

    idx_path = os.path.join(base_dir, "index.faiss")
    map_path = os.path.join(base_dir, "mapping.json")
    vec_path = os.path.join(base_dir, "vectors.npy")

    return idx_path, map_path, vec_path


def _index_paths_domain(domain: str, kb_type: str):
    """按“领域 + 类型”返回对应索引文件路径"""

    domain_key = _norm_domain_key(domain)
    base_dir = os.path.join(SEM_INDEX_ROOT, domain_key, kb_type)
    os.makedirs(base_dir, exist_ok=True)
    idx_path = os.path.join(base_dir, "index.faiss")
    map_path = os.path.join(base_dir, "mapping.json")
    vec_path = os.path.join(base_dir, "vectors.npy")
    return idx_path, map_path, vec_path


def dedup_terms_against_db(cur, terms: list[dict], project_id: int | None):
    """按 (source_term, domain) 去重，过滤已存在或本次重复的术语。"""

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


def highlight_terms(text: str, term_pairs: Iterable[tuple[str, str]]):
    """高亮术语，term_pairs = [(src, tgt), ..]"""

    if not term_pairs:
        return text

    safe = text
    for s, t in term_pairs:
        if not s:
            continue
        safe = re.sub(
            re.escape(s),
            fr"<span style='background: #fff3b0'>{s}</span>",
            safe,
            flags=re.IGNORECASE,
        )
        if t:
            safe = re.sub(
                re.escape(t),
                fr"<span style='background: #d4f6d4'>{t}</span>",
                safe,
                flags=re.IGNORECASE,
            )
    return safe


__all__ = [
    "BASE_DIR",
    "DB_PATH",
    "PROJECT_DIR",
    "SEM_INDEX_ROOT",
    "LOG_DIR",
    "LOG_FILE",
    "log_event",
    "Document",
    "KBEmbedder",
    "recommend_for_segment",
    "build_prompt_strict",
    "build_prompt_soft",
    "make_sk",
    "sk",
    "_norm_domain_key",
    "_project_domain",
    "_index_paths",
    "_index_paths_domain",
    "dedup_terms_against_db",
    "highlight_terms",
]
