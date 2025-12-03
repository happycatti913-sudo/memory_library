# 1127.py 逻辑架构图

本图概述 Streamlit 应用 `1127.py` 的核心组件及其依赖关系，按功能层次划分：配置与日志、项目与文件管理、向量语义索引、翻译流水线，以及 UI 选项卡。

```mermaid
flowchart TD
    subgraph Config[配置与基础设施]
        Paths[路径与目录常量\nBASE_DIR/DB_PATH/PROJECT_DIR/SEM_INDEX_ROOT]
        Logging[log_event\n轻量日志]
        KBImports[可选 kb_dynamic 引入\nKBEmbedder/recommend/build_prompt_*]
    end

    subgraph DataLayer[项目与数据管理]
        Projects[get_terms_for_project\nrender_index_manager]
        Files[register/fetch/remove_project_file\ncleanup_project_files]
        Corpus[import_corpus_from_upload\nensure_domain_columns_and_backfill]
        DomainMaps[_ensure_project_ref_map/_ensure_project_switch_map]
    end

    subgraph Semantic[语义向量索引]
        Embedding[get_embedder/_lazy_embedder]
        BuildIndex[build_project_vector_index\nrebuild_project_semantic_index]
        LoadSave[_load/_save_index & _index_paths]
        Search[semantic_retrieve\nalign_semantic]
        Consistency[semantic_consistency_report]
    end

    subgraph Translation[翻译与提示工程]
        InputPrep[split_blocks/split_sents\nread_docx/txt/pdf]
        Terminology[build_term_hint/check_term_consistency\nhighlight_terms]
        Instructions[build_instruction]
        DeepSeek[ds_translate/ds_extract_terms\ntranslate_block_with_kb]
        Export[build_bilingual_lines\nexport_csv_bilingual/export_docx_bilingual]
    end

    subgraph UI[Streamlit 页面]
        Tabs[Tab1 项目管理\nTab2 术语库\nTab3 历史\nTab4 语料]
        IndexMgr[render_index_manager_by_domain]
    end

    Paths --> Projects
    Paths --> LoadSave
    Logging --> Tabs
    KBImports --> Terminology
    Projects --> Files
    Projects --> BuildIndex
    Files --> InputPrep
    Corpus --> BuildIndex
    BuildIndex --> LoadSave
    Embedding --> BuildIndex
    LoadSave --> Search
    Search --> Consistency
    InputPrep --> DeepSeek
    Terminology --> DeepSeek
    Instructions --> DeepSeek
    DeepSeek --> Export
    Export --> Tabs
    IndexMgr --> Tabs
```

- **配置与基础设施**：定义基础路径、日志、可选的动态术语/向量组件，为后续模块提供依赖。【F:1127.py†L24-L120】
- **数据层**：管理项目、文件、领域/项目映射以及语料导入，支撑索引与翻译上下文构建。【F:1127.py†L214-L706】【F:1127.py†L2641-L2761】
- **语义索引**：封装向量生成、索引读写、召回与一致性报告，按领域与项目分级存储。【F:1127.py†L830-L1671】
- **翻译流水线**：涵盖文档解析、分段、术语提示与检查、指令生成、DeepSeek 请求和双语导出。【F:1127.py†L870-L2627】
- **UI 视图**：Streamlit 选项卡与索引管理面板，将上述能力组合为可交互界面。【F:1127.py†L2807-L3576】
```
