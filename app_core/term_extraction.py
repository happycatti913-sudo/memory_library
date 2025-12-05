# -*- coding: utf-8 -*-
"""术语提取与示例对齐工具。

默认使用 DeepSeek 生成式接口抽取结构化术语；当未提供 Key 或显式选择本地模式时，
会退回语料向量模型完成抽取。
"""
from __future__ import annotations

import json
import re
from typing import Iterable

import requests

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

    zh_candidates = _dedup_keep(re.findall(r"[\u4e00-\u9fa5]{2,8}", txt))
    en_candidates = _dedup_keep(
        re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}(?: [A-Za-z0-9\-]{2,}){0,2}", txt)
    )
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

    zh_to_en: dict[str, str | None] = {}
    en_to_zh: dict[str, str | None] = {}
    if zh_candidates and en_candidates:
        zh_emb = encode(zh_candidates)
        en_emb = encode(en_candidates)

        sim = zh_emb @ en_emb.T

        def _best_match(row):
            if row.size == 0:
                return None, -1.0
            idx = int(row.argmax())
            return idx, float(row[idx])

        for i, row in enumerate(sim):
            j, score = _best_match(row)
            zh_to_en[zh_candidates[i]] = en_candidates[j] if j is not None and score >= 0.35 else None

        for j, col in enumerate(sim.T):
            i, score = _best_match(col)
            en_to_zh[en_candidates[j]] = zh_candidates[i] if i is not None and score >= 0.35 else None

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
        is_zh = term in zh_candidates
        is_en = term in en_candidates

        tgt_term = term
        if tgt_lang.startswith("zh") and is_en:
            tgt_term = en_to_zh.get(term)
        elif tgt_lang.startswith("en") and is_zh:
            tgt_term = zh_to_en.get(term)

        out.append(
            {
                "source_term": term if src_lang.startswith("zh") and is_zh else (en_to_zh.get(term) if is_en else None),
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
    """使用 DeepSeek 抽取术语；可选回退至本地语料模型。

    当 ``prefer_corpus_model`` 为真或未提供 ``ak`` 时，直接走本地语料模型；
    否则调用 DeepSeek Chat 生成结构化术语，解析 JSON 后返回。
    """

    txt = (text or "").strip()
    if not txt:
        if st:
            st.warning("输入语料为空，请输入包含术语的文本（不少于 1-2 个词）。")
        return []

    if prefer_corpus_model or not ak:
        if not ak and st and not prefer_corpus_model:
            st.info("未检测到 DeepSeek Key，将直接使用语料库同款模型做结构化术语抽取。")
        return extract_terms_with_corpus_model(
            txt,
            max_terms=30,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            default_domain=default_domain,
        )

    system_msg = (
        "You are a bilingual terminology mining assistant."
        "Only return structured terminology entries in JSON."
    )
    user_msg = f"""
Source language: {src_lang}
Target language: {tgt_lang}
任务:从给定文本中抽取双语术语条目，输出 JSON 数组。字段名与取值必须是中文。
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
{txt}
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
        txt_out = data["choices"][0]["message"]["content"].strip()
        start = txt_out.find("[")
        end = txt_out.rfind("]")
        if start == -1 or end == -1:
            raise ValueError("未找到 JSON 数组")
        arr = json.loads(txt_out[start : end + 1])
        out = []
        for o in arr:
            src = (o.get("source_term") or o.get("source") or "").strip()
            tgt = (o.get("target_term") or o.get("target") or "").strip()
            dom = (o.get("domain") or "").strip() or (default_domain or None)
            strat = (o.get("strategy") or "").strip() or None
            ex = (o.get("example") or "").strip() or None
            if src:
                out.append(
                    {
                        "source_term": src,
                        "target_term": tgt,
                        "domain": dom,
                        "strategy": strat,
                        "example": ex,
                    }
                )
        return out
    except Exception:
        # 回退到本地语料模型，避免 DeepSeek 调用失败时完全无结果
        return extract_terms_with_corpus_model(
            txt,
            max_terms=30,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            default_domain=default_domain,
        )
