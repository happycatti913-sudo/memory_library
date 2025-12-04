# -*- coding: utf-8 -*-
"""术语提取与示例对齐工具（本地向量版，仅用语料向量，不调用外部API）。"""
from __future__ import annotations

import re
from typing import Iterable

try:  # 仅用于 UI 友好提示，不硬性依赖
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - CLI/测试环境
    st = None

from app_core.semantic_index import get_embedder
from app_core.text_utils import _norm_text, split_sents


def _split_sentences_for_terms(text: str) -> list[str]:
    """轻量分句，兼容中英文标点。"""
    if not text:
        return []
    txt = _norm_text(text)
    if not txt:
        return []
    parts = re.split(r"(?<=[。！？；.!?])\s+|\n+", txt)
    return [p.strip() for p in parts if p.strip()]


def _locate_example_pair(example: str | None, src_full: str | None, tgt_full: str | None):
    """在双语文本中为例句找到对应译文。"""
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
    """使用语料向量模型抽取术语并生成结构化条目。"""
    txt = (text or "").strip()
    if not txt:
        if st:
            st.warning("输入语料为空，请输入包含术语的文本（不少于 1-2 个词）。")
        return []

    backend, encode = get_embedder()

    def _dedup_keep(seq: Iterable[str]):
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    zh_candidates = re.findall(r"[\u4e00-\u9fa5]{2,8}", txt)
    en_candidates = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}(?: [A-Za-z0-9\-]{2,}){0,2}", txt)
    candidates = _dedup_keep(zh_candidates + en_candidates)
    if not candidates:
        if st:
            st.info(
                "未找到满足正则的术语候选，请检查文本是否只包含数字/符号，或术语长度不在 2-8 字、3-15 字符范围内。"
            )
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
        return sents[0] if sents else None

    out = []
    for term, _ in ranked:
        ex = _example_for(term)
        tgt_term = term
        if tgt_lang.startswith("zh"):
            tgt_term = term
        elif tgt_lang.startswith("en"):
            tgt_term = term

        out.append(
            {
                "source_term": term if src_lang.startswith("zh") else None,
                "target_term": tgt_term,
                "domain": default_domain or None,
                "strategy": None,
                "example": ex,
                "score": float(_),
            }
        )
    return out


def ds_extract_terms(
    text: str,
    ak: str,
    model: str,
    *,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    default_domain: str | None = None,
    prefer_corpus_model: bool | None = None,
    **kwargs,
):
    """仅使用本地语料向量模型抽取术语（DeepSeek 禁用）。"""
    return extract_terms_with_corpus_model(
        text or "",
        max_terms=30,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        default_domain=default_domain,
    )
