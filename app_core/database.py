"""SQLite 连接与表结构初始化。"""

import sqlite3

import streamlit as st

from .config import DB_PATH, log_event


def ensure_domain_columns_and_backfill(conn, cur, corpus_table: str = "corpus"):
    """确保 items / corpus 表存在 domain 列，并用项目领域回填语料。"""

    cols = [r[1] for r in cur.execute("PRAGMA table_info(items)").fetchall()]
    if "domain" not in cols:
        cur.execute("ALTER TABLE items ADD COLUMN domain TEXT;")
        conn.commit()

    cols = [r[1] for r in cur.execute(f"PRAGMA table_info({corpus_table})").fetchall()]
    if "domain" not in cols:
        cur.execute(f"ALTER TABLE {corpus_table} ADD COLUMN domain TEXT;")
        conn.commit()

    try:
        cur.execute(f"""
            UPDATE {corpus_table}
            SET domain = (
              SELECT i.domain FROM items i WHERE i.id = {corpus_table}.project_id
            )
            WHERE domain IS NULL AND project_id IS NOT NULL;
        """)
        conn.commit()
    except Exception:
        pass


def _has_col(cur, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())


def ensure_col(conn, cur, table: str, col: str, col_type: str):
    """确保指定表存在某列；如不存在则添加并立即提交。"""

    cur.execute(f"PRAGMA table_info({table})")
    cols = {r[1] for r in cur.fetchall()}
    if col not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        conn.commit()


def _get_domain_for_proj(cur, project_id: int):
    """根据项目ID读取 items.domain; 若不存在或为空, 返回 None。"""

    try:
        row = cur.execute(
            "SELECT domain FROM items WHERE id=?",
            (int(project_id),),
        ).fetchone()
        if not row:
            return None
        val = (row[0] or "").strip()
        return val or None
    except Exception:
        return None


def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        body TEXT,
        tags TEXT,
        domain TEXT,
        type TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS item_ext (
        id INTEGER PRIMARY KEY,
        item_id INTEGER,
        src_path TEXT,
        FOREIGN KEY(item_id) REFERENCES items(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS project_files (
        id INTEGER PRIMARY KEY,
        project_id INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        file_name TEXT,
        uploaded_at TEXT DEFAULT (datetime('now')),
        note TEXT,
        FOREIGN KEY(project_id) REFERENCES items(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS term_ext (
        id INTEGER PRIMARY KEY,
        source_term TEXT NOT NULL,
        target_term TEXT,
        domain TEXT,
        project_id INTEGER,
        strategy TEXT,
        example TEXT,
        category TEXT,
        FOREIGN KEY(project_id) REFERENCES items(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trans_ext (
        id INTEGER PRIMARY KEY,
        project_id INTEGER,
        src_text TEXT,
        tgt_text TEXT,
        mode TEXT,
        segment_count INTEGER,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(project_id) REFERENCES items(id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS corpus (
        id INTEGER PRIMARY KEY,
        project_id INTEGER,
        text TEXT,
        lang TEXT,
        source TEXT,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(project_id) REFERENCES items(id)
    );
    """)
    conn.commit()

    # Ensure domain columns/backfill after base tables exist.
    ensure_domain_columns_and_backfill(conn, cur, corpus_table="corpus")

    for t, cols in {
        "items": [
            ("type", "TEXT"),
            ("tags", "TEXT"),
            ("scene", "TEXT"),
            ("prompt", "TEXT"),
            ("mode", "TEXT"),
            ("body", "TEXT"),
            ("created_at", "TEXT"),
            ("updated_at", "TEXT"),
            ("trans_type", "TEXT"),
        ],
        "item_ext": [("src_path", "TEXT")],
        "term_ext": [
            ("domain", "TEXT"),
            ("project_id", "INTEGER"),
            ("strategy", "TEXT"),
            ("example", "TEXT"),
            ("note", "TEXT"),
            ("category", "TEXT"),
        ],
        "trans_ext": [
            ("src_path", "TEXT"),
            ("output_text", "TEXT"),
            ("lang_pair", "TEXT"),
            ("stats_json", "TEXT"),
            ("segments", "INTEGER"),
            ("term_hit_total", "INTEGER"),
        ],
        "corpus": [
            ("title", "TEXT"),
            ("project_id", "INTEGER"),
            ("lang_pair", "TEXT"),
            ("src_text", "TEXT"),
            ("tgt_text", "TEXT"),
            ("note", "TEXT"),
            ("created_at", "TEXT"),
            ("domain", "TEXT"),
            ("source", "TEXT"),
        ],
    }.items():
        for c, tp in cols:
            ensure_col(conn, cur, t, c, tp)

    cur.execute("UPDATE items SET type='project' WHERE IFNULL(type,'')='' ")
    cur.execute(
        "UPDATE items SET created_at = COALESCE(created_at, strftime('%Y-%m-%d %H:%M:%S','now'))"
    )
    conn.commit()
    ensure_col(conn, cur, "term_ext", "example_vector_id", "INTEGER")

    try:
        cur.execute("PRAGMA table_info(term_ext)")
        cols = [c[1] for c in cur.fetchall()]
        if "project_id" not in cols:
            cur.execute("ALTER TABLE term_ext ADD COLUMN project_id INTEGER")
            conn.commit()
    except Exception as e:
        st.warning(f"术语表结构检查:{e}")

    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_term_ext_project ON term_ext(project_id)")
        conn.commit()
    except Exception as e:
        log_event("WARNING", "索引创建跳过", error=str(e))

    return conn, cur


__all__ = [
    "init_db",
    "ensure_domain_columns_and_backfill",
    "_get_domain_for_proj",
    "_has_col",
    "ensure_col",
]
