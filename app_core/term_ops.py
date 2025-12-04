# -*- coding: utf-8 -*-
"""术语加载与一致性检查等通用逻辑。"""

from __future__ import annotations

from typing import Any


def get_terms_for_project(cur, pid: int, use_dynamic: bool = True):
    """统一术语加载接口，返回术语映射与元信息列表。"""
    rows_static = cur.execute(
        """
        SELECT source_term, target_term, domain
        FROM term_ext
        WHERE project_id = ?
        """,
        (pid,),
    ).fetchall()

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

    dedup: dict[tuple[str, str], tuple[str, str, str, str, int | None]] = {}

    for s, t, d in rows_static:
        if not s:
            continue
        s_raw = (s or "").strip()
        t_raw = (t or "").strip()
        d_raw = (d or "").strip()
        key = (s_raw.lower(), d_raw.lower())
        if key not in dedup:
            dedup[key] = (s_raw, t_raw, d_raw, "static", int(pid))

    for row in rows_dynamic:
        if len(row) == 4:
            s, t, d, pid_term = row
        else:
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

    term_map: dict[str, str] = {}
    term_meta: list[dict[str, Any]] = []

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


def check_term_consistency(out_text: str, term_dict: dict, source_text: str = "") -> list[str]:
    """粗略检查译文中可能未遵守的术语。"""
    if not out_text or not term_dict:
        return []
    warns = []
    s = (source_text or "")[:2000]
    out_low = out_text.lower()
    for k, v in term_dict.items():
        if not k or not v:
            continue
        hit_src = (k.lower() in s.lower()) if any(ord(ch) < 128 for ch in k) else (k in s)
        if hit_src:
            ok = (v.lower() in out_low) if any(ord(ch) < 128 for ch in v) else (v in out_text)
            if not ok:
                warns.append(f"{k}→{v}")
    return warns
