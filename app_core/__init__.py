"""app_core 包：承载从 1127.py 拆分出来的核心模块。

此文件显式声明可用的子模块，方便调用方使用静态分析或自动补全，
同时避免隐式导入带来的运行时开销。
"""

# 模块名称清单，避免 linters 将文件视为空文件。
__all__ = [
    "config",
    "corpus_ops",
    "database",
    "file_ops",
    "projects",
    "semantic_index",
    "semantic_ops",
    "term_extraction",
    "term_ops",
    "text_utils",
    "translation_ops",
    "ui_common",
    "ui_corpus",
    "ui_history",
    "ui_index",
    "ui_projects",
    "ui_terms",
]