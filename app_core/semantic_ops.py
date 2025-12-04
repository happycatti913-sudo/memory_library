# -*- coding: utf-8 -*-
"""语义召回、对齐与一致性检查相关的工具函数。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .semantic_index import _lazy_import_vec, _load_index, get_embedder


def semantic_retrieve(
    project_id: int,
    query_text: str,
    topk: int = 20,
    scope: str = "project",
    min_char: int = 3,
    cur=None,
):
    """语料库语义召回(句级，中英句对)。"""

    q = (query_text or "").strip()
    if len(q) < min_char:
        return []

    def _split(text: str) -> list[str]:
        try:
            if "split_sents" in globals():
                from app_core.text_utils import split_sents

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

    try:
        backend, encode = get_embedder()
    except RuntimeError as e:
        st.error(f"向量模型加载失败: {e}")
        return []

    mode, index, mapping, vecs = _load_index(int(project_id), cur)
    if mode == "none" or not mapping:
        return []

    np_mod, faiss, *_ = _lazy_import_vec()
    np = np_mod

    cur_domain = None
    if scope == "domain" and cur is not None:
        try:
            from app_core.database import _get_domain_for_proj

            cur_domain = _get_domain_for_proj(cur, int(project_id))
        except Exception:
            cur_domain = None

    def _scope_ok(meta: dict) -> bool:
        if scope == "project":
            return int(meta.get("project_id", 0) or 0) == int(project_id)
        if scope == "domain" and cur_domain:
            return (meta.get("domain") or "") == (cur_domain or "")
        return True

    all_hits: list[tuple[float, dict, str, str]] = []
    per_piece_k = max(topk * 3, topk)

    for piece in pieces:
        if not piece:
            continue
        try:
            qv = encode([piece])
            qv = qv.reshape(-1)
            if qv.shape[0] == 0:
                continue

            if mode == "faiss" and faiss is not None and index is not None:
                if faiss is None:
                    continue
                D, I = index.search(qv.reshape(1, -1).astype("float32"), per_piece_k)
                for score, idx in zip(D.reshape(-1), I.reshape(-1)):
                    idx = int(idx)
                    if idx < 0 or idx >= len(mapping):
                        continue
                    meta = mapping[idx] or {}
                    if not isinstance(meta, dict) or not _scope_ok(meta):
                        continue
                    src_sent = (meta.get("src") or "").strip()
                    tgt_sent = (meta.get("tgt") or "").strip()
                    if not src_sent and not tgt_sent:
                        continue
                    all_hits.append((float(score), meta, src_sent, tgt_sent))

            elif mode == "fallback" and vecs is not None:
                arr = np.asarray(vecs, dtype="float32")
                if arr.ndim != 2 or arr.shape[0] == 0:
                    continue
                sims = arr @ qv.reshape(-1, 1)
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
                    if not isinstance(meta, dict) or not _scope_ok(meta):
                        continue
                    src_sent = (meta.get("src") or "").strip()
                    tgt_sent = (meta.get("tgt") or "").strip()
                    if not src_sent and not tgt_sent:
                        continue
                    all_hits.append((score, meta, src_sent, tgt_sent))
        except Exception:
            continue

    if not all_hits:
        return []

    dedup = {}
    for score, meta, src_sent, tgt_sent in all_hits:
        key = (src_sent, tgt_sent)
        if key not in dedup or score > dedup[key][0]:
            dedup[key] = (score, meta, src_sent, tgt_sent)

    hits = sorted(dedup.values(), key=lambda x: x[0], reverse=True)
    return hits[:topk]


def semantic_consistency_report(
    project_id: int,
    blocks_src: list,
    blocks_tgt: list,
    term_map: dict,
    topk: int = 3,
    thr: float = 0.70,
    cur=None,
):
    """译后一致性报告 (语义 + 术语，按段落)。"""

    hits_all = []
    n = min(len(blocks_src or []), len(blocks_tgt or []))
    if n == 0:
        return pd.DataFrame([])

    for i, (s, t) in enumerate(zip(blocks_src[:n], blocks_tgt[:n]), 1):
        s = s or ""
        t = t or ""

        try:
            hits = semantic_retrieve(project_id, t, topk=topk, cur=cur)
        except Exception:
            hits = []

        top_score = float(hits[0][0]) if hits else 0.0

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


def _build_ref_context(
    project_id: int,
    query_text: str,
    topk: int = 20,
    min_sim: float = 0.25,
    prefer_side: str = "both",
    scope: str = "project",
    top_n: int = 5,
    cur=None,
) -> str:
    """构建参考例句(句级，中英对照)。"""

    try:
        hits = semantic_retrieve(
            project_id,
            query_text,
            topk=topk,
            scope=scope,
            cur=cur,
        )
    except Exception as e:
        try:
            st.warning(f"参考检索失败: {e}")
        except Exception:
            pass
        return ""

    if not hits:
        return ""

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

        if not ch and not en:
            continue

        key = (ch, en)
        if key in seen:
            continue
        seen.add(key)

        selected.append((score, meta, ch, en))
        if len(selected) >= top_n:
            break

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

    ctx_lines = ["参考例句(用于保持术语与风格一致):"]
    for idx, (sc, meta, ch, en) in enumerate(selected, 1):
        dom = (meta.get("domain") or "").strip() if isinstance(meta, dict) else ""
        title = (meta.get("title") or "").strip() if isinstance(meta, dict) else ""
        tag_info = " · ".join(x for x in [dom, title] if x)

        ch_show = ch.replace("\n", " ").strip()
        en_show = en.replace("\n", " ").strip()

        if tag_info:
            ctx_lines.append(f"[{idx} · {tag_info} · sim={sc:.2f}] {ch_show}\n{en_show}")
        else:
            ctx_lines.append(f"[{idx} · sim={sc:.2f}] {ch_show}\n{en_show}")

    return "\n".join(ctx_lines)


def _lazy_embedder():
    try:
        from sentence_transformers import SentenceTransformer

        mdl = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

        def _emb(texts):
            arr = mdl.encode(texts, normalize_embeddings=True)
            return arr.astype("float32")

        return _emb, "sbert"
    except Exception:
        def _emb(texts):
            vec = TfidfVectorizer(min_df=1).fit_transform(texts)
            norms = np.sqrt((vec.multiply(vec)).sum(axis=1)).A.ravel() + 1e-8
            vec = vec.multiply(1 / norms[:, None])
            return vec

        return _emb, "tfidf"


def align_semantic(src_sents, tgt_sents, max_jump=3):
    """简单贪心 + 滑窗的 1-1 句对齐.返回 [(src, tgt, score)]"""

    if not src_sents or not tgt_sents:
        return []

    emb, kind = _lazy_embedder()

    if kind == "sbert":
        E1 = emb(src_sents)
        E2 = emb(tgt_sents)
        sims = E1 @ E2.T
    else:
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 2),
            min_df=1,
        )
        combo = src_sents + tgt_sents
        X = vec.fit_transform(combo)
        n = len(src_sents)
        E1 = X[:n, :]
        E2 = X[n:, :]
        sims = cosine_similarity(E1, E2, dense_output=True)

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

