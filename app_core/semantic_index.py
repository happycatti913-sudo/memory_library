# -*- coding: utf-8 -*-
"""语义索引与向量化工具。"""
from __future__ import annotations

import json
import os
from typing import Callable

import streamlit as st

from .config import _index_paths, _index_paths_domain, log_event


def _lazy_import_vec():
    """懒加载向量相关依赖，允许缺失 faiss/fastembed。"""
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
        from fastembed import TextEmbedding as FastEmbedModel  # noqa: F401
    except Exception:
        FastEmbedModel = None

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
    except Exception:
        TfidfVectorizer = None

    return np, faiss, SentenceTransformer, FastEmbedModel, TfidfVectorizer, None


@st.cache_resource(show_spinner=False)
def get_embedder():
    """返回 (backend, encode) 元组，仅使用 SentenceTransformer 句向量。"""
    import numpy as np

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:  # pragma: no cover - 环境缺失时提示
        st.error(f"❌ 无法导入 sentence-transformers，请先安装依赖: {e}")
        raise RuntimeError("sentence-transformers not available") from e

    model_name = "distiluse-base-multilingual-cased-v1"

    try:
        model = SentenceTransformer(model_name)
    except Exception as e:  # pragma: no cover - 运行时加载失败
        st.error(f"❌ 加载句向量模型 {model_name} 失败: {e}")
        raise RuntimeError(f"failed to load sentence transformer model {model_name}") from e

    def encode_st(texts: list[str]):
        if not texts:
            dim = model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype="float32")

        emb = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            convert_to_numpy=True,
        ).astype("float32")

        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        return (emb / norms).astype("float32")

    st.info(f"✅ 已启用 SentenceTransformer 句向量: {model_name}")
    return "st", encode_st


def _load_index(project_id: int, cur=None):
    np, faiss, *_ = _lazy_import_vec()
    idx_path, map_path, vec_path = _index_paths(project_id, cur)
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

    log_event(
        "WARNING",
        "semantic index not found",
        project_id=project_id,
        idx_path=idx_path,
        vec_path=vec_path,
    )
    return "none", None, mapping, None


def _save_index(project_id: int, mode: str, index, mapping, vecs=None, cur=None):
    np, faiss, *_ = _lazy_import_vec()
    idx_path, map_path, vec_path = _index_paths(project_id, cur)
    if mode == "faiss" and index is not None:
        faiss.write_index(index, idx_path)
    elif mode == "fallback" and vecs is not None:
        np.save(vec_path, vecs.astype("float32"))
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def _load_index_domain(domain: str, kb_type: str):
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
        if os.path.exists(vec_path):
            try:
                os.remove(vec_path)
            except OSError:
                pass
    elif mode == "fallback" and vecs is not None:
        np.save(vec_path, vecs.astype("float32"))
        if os.path.exists(idx_path):
            try:
                os.remove(idx_path)
            except OSError:
                pass
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def build_strategy_index_for_domain(cur, domain: str):
    """为指定领域重建【翻译策略】单语索引。"""
    import numpy as _np

    np, faiss, *_ = _lazy_import_vec()
    backend, encode = get_embedder()

    dom = (domain or "").strip() or "未分类"

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

    rows = cur.execute(
        """
        SELECT id,
               IFNULL(domain,''), IFNULL(title,''), content,
               IFNULL(collection,''), IFNULL(source,'')
          FROM strategy_texts
         WHERE IFNULL(domain,'') = ?
         ORDER BY id ASC
        """,
        (dom,),
    ).fetchall()

    texts, metas = [], []
    for sid, d, ttl, content, coll, src in rows:
        txt = (content or "").strip()
        if not txt:
            continue
        texts.append(txt)
        metas.append(
            {
                "strategy_id": sid,
                "domain": d or dom,
                "title": ttl,
                "content_preview": txt[:200],
                "collection": coll,
                "source": src,
                "kb_type": "strategy",
            }
        )

    if not texts:
        _save_index_domain(dom, "strategy", "none", None, [])
        return {"added": 0, "total": 0}

    new_vecs = encode(texts)
    if hasattr(new_vecs, "toarray"):
        new_vecs = new_vecs.toarray()
    new_vecs = _np.asarray(new_vecs, dtype="float32")
    if new_vecs.ndim == 1:
        new_vecs = new_vecs.reshape(1, -1)
    new_vecs = new_vecs / (_np.linalg.norm(new_vecs, axis=1, keepdims=True) + 1e-12)

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


def build_project_vector_index(
    cur,
    project_id: int,
    *,
    use_src: bool = True,
    use_tgt: bool = True,
    split_fn: Callable[[str, str], tuple[list[str], list[str]]] | None = None,
):
    """为指定项目所属领域重建向量索引(句对版，中英对照)。"""
    import numpy as _np

    np, faiss, *_ = _lazy_import_vec()
    backend, encode = get_embedder()

    pid = int(project_id)

    proj_domain = None
    try:
        row = cur.execute(
            "SELECT IFNULL(domain,'') FROM items WHERE id=?",
            (pid,),
        ).fetchone()
        if row:
            proj_domain = (row[0] or "").strip()
    except Exception:
        proj_domain = None

    if not proj_domain:
        proj_domain = "未分类"

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
        (proj_domain,),
    ).fetchall()

    texts, metas = [], []

    for cid, s, t, ttl, lp, pj, dom in rows:
        s = (s or "").strip()
        t = (t or "").strip()
        if not s and not t:
            continue

        if split_fn is not None:
            try:
                src_sents, tgt_sents = split_fn(s, t)
            except Exception:
                src_sents, tgt_sents = [], []
        else:
            try:
                src_sents = split_sents(s, lang_hint="zh") if "split_sents" in globals() else (s.split("。") if s else [])
                tgt_sents = split_sents(t, lang_hint="en") if "split_sents" in globals() else (t.split(".") if t else [])
            except Exception:
                src_sents = (s.split("。") if s else [])
                tgt_sents = (t.split(".") if t else [])

        n = min(len(src_sents), len(tgt_sents)) if (use_src and use_tgt) else len(src_sents or [])

        for idx in range(n):
            src_j = (src_sents[idx] if idx < len(src_sents) else "").strip()
            tgt_j = (tgt_sents[idx] if idx < len(tgt_sents) else "").strip()
            if not src_j:
                continue

            texts.append(src_j)
            metas.append(
                {
                    "corpus_id": cid,
                    "idx": idx,
                    "src": src_j,
                    "tgt": tgt_j,
                    "project_id": pj,
                    "domain": dom or proj_domain or "",
                    "title": ttl,
                    "lang_pair": lp or "",
                    "kb_type": "bilingual",
                }
            )

    if not texts:
        try:
            _save_index(pid, "none", None, [], vecs=None, cur=cur)
        except Exception:
            pass
        return {"added": 0, "total": 0}

    new_vecs = encode(texts)
    if hasattr(new_vecs, "toarray"):
        new_vecs = new_vecs.toarray()
    new_vecs = _np.asarray(new_vecs, dtype="float32")
    if new_vecs.ndim == 1:
        new_vecs = new_vecs.reshape(1, -1)
    new_vecs = new_vecs / (_np.linalg.norm(new_vecs, axis=1, keepdims=True) + 1e-12)

    if faiss is not None and backend in ("st", "fastembed"):
        dim = int(new_vecs.shape[1])
        index = faiss.IndexFlatIP(dim)
        index.add(new_vecs)
        _save_index(pid, "faiss", index, metas, cur=cur)
        total = int(index.ntotal)
    else:
        vecs = new_vecs
        _save_index(pid, "fallback", None, metas, vecs=vecs, cur=cur)
        total = int(vecs.shape[0])

    return {"added": len(texts), "total": total}


def rebuild_project_semantic_index(cur, project_id: int, *, split_fn: Callable[[str, str], tuple[list[str], list[str]]] | None = None) -> dict:
    """对外的重建语义索引入口函数。"""
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return {"ok": False, "added": 0, "total": 0, "msg": f"非法项目ID: {project_id!r}"}

    try:
        res = build_project_vector_index(cur, pid, use_src=True, use_tgt=True, split_fn=split_fn)
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
