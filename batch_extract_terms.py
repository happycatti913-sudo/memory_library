#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch-extract glossary terms from the `corpus` table using the same DeepSeek
prompt/logic as 1127.py (_extract_terms_for_upload + ds_extract_terms), but
without Streamlit. Results are written to term_ext.

Usage examples (run inside e:\\齐译):
  .\\.venv2\\Scripts\\python.exe batch_extract_terms.py --note nejm_full_import --limit 200
  .\\.venv2\\Scripts\\python.exe batch_extract_terms.py --lang zh --batch-chars 6000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tomllib

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "kb.db"
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
DEFAULT_STATE = BASE_DIR / ".batch_extract_terms.state"


# Small row helper to tolerate missing keys / None
def rget(row, key, default=None):
    try:
        val = row[key]
    except Exception:
        return default
    return val if val is not None else default


# ---------------- DeepSeek glue (copied from 1127.py) ----------------
def load_deepseek():
    """Load DeepSeek API key/model from .streamlit/secrets.toml."""
    if not SECRETS_PATH.exists():
        raise RuntimeError(f"secrets.toml not found: {SECRETS_PATH}")
    data = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    try:
        ds = data["deepseek"]
        return ds["api_key"], ds.get("model", "deepseek-chat")
    except Exception as e:
        raise RuntimeError("deepseek.api_key missing in secrets.toml") from e


def _make_http():
    s = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def ds_extract_terms(
    text: str,
    ak: str,
    model: str,
    src_lang: str = "zh",
    tgt_lang: str = "en",
    timeout: int = 120,
):
    """
    Same prompt/logic as 1127.py: send text to DeepSeek and parse JSON array.
    """
    system_msg = (
        "You are a terminology mining assistant. Extract high-value bilingual term pairs "
        "suitable for a project glossary. Return JSON array only. No extra text."
    )
    user_msg = f"""
Source language: {src_lang}
Target language: {tgt_lang}
任务:从给定文本中抽取双语术语条目.输出 JSON 数组。字段名与取值必须是中文。
字段定义:
- source_term: 源语(中文术语或专名)
- target_term: 译文(英文)
- target_term_alt: 译文备选(英文, 与 target_term 不同)
- domain: 领域.取值集合之一:["医学"]
- strategy: 译法策略.取值集合之一:["直译","意译","转译","音译","省译","增译","规范化","其它"]
- example: 例句(原文中包含该术语的一句.尽量保留标点)

要求:
1) 仅输出 JSON.不要额外说明。
2) 同一术语重复时合并.选择最典型的例句。
3) 若无法判断 domain/strategy.填“其它”。
4) 不要限制数量，尽量多抽取高价值术语。

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
    session = _make_http()
    r = session.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    txt = data["choices"][0]["message"]["content"].strip()

    def _extract_array(raw: str) -> str | None:
        cleaned = raw
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0]
        m = re.search(r"\[[\s\S]*\]", cleaned)
        if m:
            return m.group(0)
        m = re.search(r"\[[\s\S]*\]", raw)
        return m.group(0) if m else None

    arr_txt = _extract_array(txt)
    if not arr_txt:
        err_dir = BASE_DIR / "deepseek_errors"
        err_dir.mkdir(exist_ok=True)
        fname = err_dir / f"fail_{int(time.time())}.txt"
        try:
            fname.write_text(txt, encoding="utf-8")
        except Exception:
            pass
        raise RuntimeError(f"DeepSeek response missing JSON array (saved to {fname.name})")
    try:
        arr = json.loads(arr_txt)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"DeepSeek JSON parse failed: {e}") from e

    cleaned = []
    for o in arr:
        src = (o.get("source_term") or o.get("source") or "").strip()
        tgt = (o.get("target_term") or o.get("target") or "").strip()
        tgt_alt = (o.get("target_term_alt") or o.get("alt") or "").strip()
        dom = (o.get("domain") or o.get("领域") or "").strip()
        strategy = (o.get("strategy") or o.get("策略") or "").strip()
        example = (o.get("example") or o.get("例句") or "").strip()
        if not src:
            continue
        cleaned.append(
            {
                "source_term": src,
                "target_term": tgt,
                "target_term_alt": tgt_alt,
                "domain": dom or None,
                "strategy": strategy or None,
                "example": example or None,
            }
        )
    return cleaned


# ---------------- DB helpers ----------------
def open_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_term_ext(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS term_ext (
            id INTEGER PRIMARY KEY,
            source_term TEXT NOT NULL,
            target_term TEXT,
            domain TEXT,
            project_id INTEGER,
            strategy TEXT,
            example TEXT,
            example_vector_id INTEGER,
            category TEXT
        )
        """
    )
    conn.commit()  # ensure table exists before index attempts

    try:
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_term_proj
            ON term_ext(LOWER(TRIM(source_term)), IFNULL(project_id,-1))
            """
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # duplicates already present; skip enforcing uniqueness to avoid aborting the run
        conn.rollback()
        print("WARN: ux_term_proj unique index not enforced (duplicates exist); continuing without it.")
    try:
        conn.commit()
    except sqlite3.IntegrityError:
        # duplicates already present; skip enforcing uniqueness to avoid aborting the run
        conn.rollback()
        print("WARN: ux_term_proj unique index not enforced (duplicates exist); continuing without it.")
    else:
        return

    conn.commit()


def _is_zh(text: str) -> bool:
    zh_hits = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    en_hits = len(re.findall(r"[A-Za-z]", text or ""))
    return zh_hits >= en_hits


def pick_lang_pair(row: sqlite3.Row, text: str):
    lp = (rget(row, "lang_pair", "") or "").strip()
    if lp.startswith("英"):
        return "en", "zh"
    if lp.startswith("中"):
        return "zh", "en"
    return ("zh", "en") if _is_zh(text) else ("en", "zh")


def chunk_rows(rows, max_chars: int):
    buf = []
    size = 0
    for r in rows:
        txt = (r["src_text"] or "") + "\n" + (r["tgt_text"] or "")
        tlen = len(txt)
        if size + tlen > max_chars and buf:
            yield buf
            buf = []
            size = 0
        buf.append(r)
        size += tlen
    if buf:
        yield buf


# ---------------- main flow ----------------
def main():
    ap = argparse.ArgumentParser(description="Batch term extraction from corpus -> term_ext.")
    ap.add_argument("--note", help="only rows whose note LIKE this prefix (e.g., nejm_full_import)")
    ap.add_argument("--limit", type=int, help="max rows to process")
    ap.add_argument("--batch-chars", type=int, default=8000, help="max characters per DeepSeek call")
    ap.add_argument("--lang", choices=["zh", "en", "auto"], default="auto", help="override source language")
    ap.add_argument("--project-id", type=int, help="force project_id on inserted terms")
    ap.add_argument("--resume", action="store_true", help="resume from last processed corpus id (uses state file)")
    ap.add_argument("--state-file", type=Path, default=DEFAULT_STATE, help="state file for --resume (stores last_id)")
    args = ap.parse_args()

    ak, model = load_deepseek()
    conn = open_db()
    ensure_term_ext(conn)
    cur = conn.cursor()

    last_id = load_checkpoint(args.state_file) if args.resume else 0

    sql = "SELECT id, project_id, src_text, tgt_text, lang_pair, note, domain FROM corpus WHERE 1=1"
    params = []
    if args.note:
        sql += " AND IFNULL(note,'') LIKE ?"
        params.append(f"{args.note}%")
    if args.resume and last_id > 0:
        sql += " AND id > ?"
        params.append(last_id)
    if args.limit:
        sql += " LIMIT ?"
        params.append(args.limit)

    rows = cur.execute(sql, params).fetchall()
    print(f"loaded {len(rows)} corpus rows")
    inserted = 0
    max_id_seen = last_id

    for group in chunk_rows(rows, args.batch_chars):
        texts = []
        domains = []
        pid = args.project_id
        # choose lang once per batch
        sample_txt = (rget(group[0], "src_text", "") or "") + (rget(group[0], "tgt_text", "") or "")
        src_lang, tgt_lang = pick_lang_pair(group[0], sample_txt)
        if args.lang != "auto":
            src_lang, tgt_lang = (args.lang, "en" if args.lang == "zh" else "zh")

        for r in group:
            texts.append(rget(r, "src_text", "") or rget(r, "tgt_text", "") or "")
            domains.append(rget(r, "domain"))
            if pid is None:
                pid = rget(r, "project_id")
        big = "\n".join([t for t in texts if t]).strip()
        if not big:
            continue

        try:
            terms = ds_extract_terms(big, ak, model, src_lang=src_lang, tgt_lang=tgt_lang)
        except Exception as e:
            print(f"[skip batch] DeepSeek error: {e}")
            continue
        if not terms:
            continue

        max_id_seen = max(max_id_seen, max(r["id"] for r in group if "id" in r.keys()))

        # simple heuristic: use the most common domain in this batch
        dom = None
        if domains:
            dom = max(domains, key=lambda d: domains.count(d)) if any(domains) else None

        for t in terms:
            row = (
                t["source_term"],
                t["target_term"] or None,
                dom or t["domain"],
                pid,
                t.get("strategy") or "corpus-batch",
                t.get("example") or None,
            )
            try:
                cur.execute(
                    """
                    INSERT INTO term_ext (source_term, target_term, domain, project_id, strategy, example)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    row,
                )
                inserted += 1
            except sqlite3.IntegrityError:
                # duplicate by UNIQUE index; perform an update
                cur.execute(
                    """
                    UPDATE term_ext
                    SET target_term=COALESCE(?, target_term),
                        domain     =COALESCE(?, domain),
                        project_id =COALESCE(?, project_id),
                        strategy   =COALESCE(?, strategy),
                        example    =COALESCE(?, example)
                    WHERE LOWER(TRIM(source_term)) = LOWER(TRIM(?))
                      AND IFNULL(project_id,-1) = IFNULL(?, -1)
                    """,
                    (
                        row[1],
                        row[2],
                        row[3],
                        row[4],
                        row[5],
                        row[0],
                        row[3],
                    ),
                )
            conn.commit()
        print(f"+{len(terms)} terms (total {inserted}) [src={src_lang}->tgt={tgt_lang}]")

    if args.resume and max_id_seen > last_id:
        save_checkpoint(args.state_file, max_id_seen)
        print(f"checkpoint saved: last_id={max_id_seen} -> {args.state_file}")

    print(f"done. total inserted/updated: {inserted}")


# ---------- checkpoint helpers ----------
def load_checkpoint(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("last_id", 0))
    except Exception:
        return 0


def save_checkpoint(path: Path, last_id: int):
    try:
        path.write_text(json.dumps({"last_id": last_id}), encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
