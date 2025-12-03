# -*- coding: utf-8 -*-
"""术语提取与示例对齐工具。"""

from __future__ import annotations

import json
import re
from typing import Iterable

from app_core.semantic_index import get_embedder
from app_core.text_utils import _norm_text, split_sents


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
    """在翻译历史中为示例句找到可能的对齐译文。"""
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
        return sents[0] if sents else None

    out = []
    for term, _ in ranked:
        ex = _example_for(term)
        out.append(
            {
                "source_term": term if src_lang.startswith("zh") else None,
                "target_term": term if src_lang.startswith("en") else None,
                "domain": default_domain or None,
                "strategy": None,
                "example": ex,
                "score": float(_),
            }
        )
    return out


def ds_extract_terms(
    ak: str,
    model: str,
    text: str,
    *,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    default_domain: str | None = None,
):
    """调用 DeepSeek 生成式接口抽取结构化术语。"""
    import requests

    if not text or not text.strip():
        return []

    system_msg = (
        "You are a bilingual terminology mining assistant."
        "Only return structured terminology entries in JSON."
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
        start = txt.find("[")
        end = txt.rfind("]")
        if start == -1 or end == -1:
            return []
        arr = json.loads(txt[start : end + 1])
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
        return []
