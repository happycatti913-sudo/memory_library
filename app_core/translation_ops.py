# -*- coding: utf-8 -*-
"""翻译调用与提示构建工具。"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st

from app_core.config import log_event
from app_core.semantic_ops import _build_ref_context
from app_core.term_ops import check_term_consistency, get_terms_for_project


def get_deepseek():
    """从 Streamlit secrets 读取 DeepSeek 配置。"""
    try:
        ak = st.secrets["deepseek"]["api_key"]
        model = st.secrets["deepseek"].get("model", "deepseek-chat")
        return ak, model
    except Exception:
        return None, None


def build_term_hint(term_dict: dict, lang_pair: str, max_terms: int = 80) -> str:
    """将术语映射转成严格的 GLOSSARY 提示文本。"""
    if not term_dict or not isinstance(term_dict, dict):
        return ""

    lines: list[str] = []
    seen = set()
    items = list(term_dict.items())[: max_terms * 2]

    for src, val in items:
        if src is None:
            continue
        src = str(src).strip()
        if not src or src in seen:
            continue

        tgt, pos, note = None, None, None
        if isinstance(val, dict):
            tgt = (val.get("target") or val.get("tgt") or "").strip()
            pos = (val.get("pos") or "").strip() or None
            note = (val.get("usage_note") or val.get("note") or "").strip() or None
        elif isinstance(val, (list, tuple)) and len(val) >= 1:
            tgt = str(val[0]).strip()
            if len(val) >= 2:
                pos = str(val[1]).strip() or None
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
        return ""

    header = "GLOSSARY (STRICT):\n"
    return header + "\n".join(lines[:max_terms]) + "\n"


def build_instruction(lang_pair: str) -> str:
    """根据语向生成简洁指令，统一处理多种分隔符。"""
    lp_raw = (lang_pair or "").replace(" ", "")
    lp_norm = lp_raw.lower()
    for sep in ("→", "->", "=>", "—>", "—", "—", "—-", "——"):
        lp_norm = lp_norm.replace(sep, "-")
    lp_norm = lp_norm.replace("to", "-").replace("_", "-").replace("/", "-")

    zh_to_en_tokens = (
        "中译英",
        "中→英",
        "中->英",
        "中-英",
        "zh-en",
        "zh2en",
        "zh_en",
        "zh-en",
        "chinese-english",
        "chinese-en",
        "zh-english",
    )
    en_to_zh_tokens = (
        "英译中",
        "英→中",
        "英->中",
        "英-中",
        "en-zh",
        "en2zh",
        "en_zh",
        "en-zh",
        "english-chinese",
        "english-zh",
        "en-chinese",
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
    """使用 DeepSeek API 翻译单个文本块，带术语与可选参考。"""
    import requests

    term_hint = build_term_hint(term_dict, lang_pair)
    instr = build_instruction(lang_pair)

    if not block.strip():
        return ""

    if not term_hint:
        if term_dict:
            term_hint = (
                "GLOSSARY (STRICT):\n"
                "- Follow provided terminology exactly; do not paraphrase fixed terms.\n\n"
            )
        else:
            term_hint = (
                "GLOSSARY (STRICT):\n"
                "- Ensure consistent terminology; avoid paraphrasing fixed terms.\n\n"
            )

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

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_msg}]
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
            txt = f"[DeepSeek {resp.status_code}] {resp.text}"
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
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
            log_event(
                "ERROR",
                "DeepSeek request exception",
                error=str(e),
            )
            return f"[DeepSeek Request Error] {e}"

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
    """单段翻译主管线，整合术语加载、参考例句与一致性检查。"""
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

    term_map_all, term_meta = get_terms_for_project(cur, project_id, use_dynamic=True)

    def _detect_hits(text: str, term_map: dict[str, str]) -> dict[str, str]:
        txt_low = (text or "").lower()
        out: dict[str, str] = {}
        for k, v in (term_map or {}).items():
            if not k:
                continue
            key_low = k.lower()
            if key_low in txt_low or k in text:
                out[k] = v
        return out

    terms_in_block = _detect_hits(blk, term_map_all)

    if use_semantic:
        try:
            ref_context = _build_ref_context(
                project_id,
                blk,
                topk=20,
                min_sim=0.25,
                prefer_side="both",
                scope=scope,
                cur=cur,
            )
        except Exception:
            ref_context = ""
    else:
        ref_context = ""

    terms_corpus_dyn = _detect_hits(ref_context, term_map_all) if ref_context else {}

    terms_final = dict(terms_in_block)
    for k, v in terms_corpus_dyn.items():
        if k not in terms_final:
            terms_final[k] = v

    tgt = ds_translate(
        block=blk,
        term_dict=terms_final,
        lang_pair=lang_pair,
        ak=ak,
        model=model,
        ref_context=ref_context,
        fewshot_examples=fewshot_examples,
    )

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
