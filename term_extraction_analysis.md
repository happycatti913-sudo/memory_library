# 术语提取方式与可能失效原因

## 提取方式概览
- `app_core/term_extraction.py` 中的 `extract_terms_with_corpus_model` 使用本地语料向量模型（SentenceTransformer）为输入文本和候选术语生成向量，再按照语义相似度排序返回结构化条目。【F:app_core/term_extraction.py†L51-L120】
- 候选术语通过正则提取：
  - 中文连续 2-8 字符。
  - 英文/数字/连字符的 3-15 字符片段，允许最多两个后续词组成 1-3 词短语，能覆盖带数字或版本号的术语。【F:app_core/term_extraction.py†L78-L87】
- 模型加载依赖 `get_embedder`，固定使用 `distiluse-base-multilingual-cased-v1` 句向量模型，缺失依赖或加载失败会直接抛出 `RuntimeError` 并在界面提示错误。【F:app_core/semantic_index.py†L42-L77】
- `ds_extract_terms` 只是对 `extract_terms_with_corpus_model` 的封装，参数顺序已调整为“文本在前”，避免误把模型名或密钥当作语料；外部 DeepSeek 调用仍被禁用，因此当前唯一提取路径是本地语料模型。【F:app_core/term_extraction.py†L123-L141】【F:app_core/ui_terms.py†L690-L745】

## 无法提取术语的常见原因
1. **环境缺失或模型加载失败**：未安装 `sentence-transformers`，或模型下载/加载报错，会在 `get_embedder` 中抛出异常，导致整个提取流程提前终止，无任何术语返回。【F:app_core/semantic_index.py†L42-L59】
2. **输入文本为空或不含合法候选**：
   - 文本为空白时界面会提示“输入语料为空，请输入包含术语的文本（不少于 1-2 个词）”，随后直接返回空列表。【F:app_core/term_extraction.py†L60-L86】
   - 文本内容不包含满足正则的中英文片段时，`candidates` 为空会在界面提示“未找到满足正则的术语候选”，并返回空结果；纯数字/符号或过短/过长的片段都会被过滤。【F:app_core/term_extraction.py†L78-L87】
3. **候选范围有限导致遗漏**：
   - 中文限制在 2-8 字，单字或超长术语直接被过滤。
   - 英文正则仅支持 1-3 词的字母/连字符短语，带数字或更长的专有名词（如版本号、含符号的术语）无法进入候选，最终也无法被打分输出。【F:app_core/term_extraction.py†L78-L87】
4. **调用参数顺序出错**：此前 UI 侧将“文本、Key、模型”传给 `ds_extract_terms`，但函数参数顺序为“Key、模型、文本”，导致实际参与抽取的只有模型名（如 `deepseek-chat`），长文本被忽略；现已调整为文本优先的签名，避免再次出现这种错位。【F:app_core/term_extraction.py†L123-L141】【F:app_core/ui_history.py†L121-L144】
5. **示例句依赖简单分句**：候选术语在 `_split_sentences_for_terms` 返回的句子中找不到时，会退回首句；若输入缺乏标点导致分句失败，示例对齐可能缺失，进一步影响下游展示效果。【F:app_core/term_extraction.py†L12-L99】

如需提升提取成功率，可先确保运行环境已安装 `sentence-transformers`，并根据业务调整正则范围（允许数字/更长词组），或增加回退策略以在模型不可用时至少返回词典式抽取结果。
