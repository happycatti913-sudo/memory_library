import ast
import sys
import types
from pathlib import Path


TARGET_FUNCS = {"_norm_domain_key", "build_term_hint", "build_instruction"}


def load_helpers(extra_funcs: set[str] | None = None):
    targets = set(TARGET_FUNCS)
    if extra_funcs:
        targets.update(extra_funcs)

    module = types.ModuleType("helpers_only")

    # 先从主应用文件提取可用函数
    main_source = Path(__file__).parent.joinpath("1127.py").read_text(encoding="utf-8")
    main_tree = ast.parse(main_source)
    main_body = [
        node
        for node in main_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in targets
    ]

    helper_ast = ast.Module(body=main_body, type_ignores=[])
    helper_ast = ast.fix_missing_locations(helper_ast)
    exec(compile(helper_ast, "1127.py", "exec"), module.__dict__)

    # 再从拆分出来的 config 模块补齐缺失函数
    remaining = targets.difference(module.__dict__)
    if remaining:
        cfg_source = Path(__file__).parent.joinpath("app_core", "config.py").read_text(encoding="utf-8")
        cfg_tree = ast.parse(cfg_source)
        cfg_body = [
            node
            for node in cfg_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in remaining
        ]
        if cfg_body:
            cfg_ast = ast.Module(body=cfg_body, type_ignores=[])
            cfg_ast = ast.fix_missing_locations(cfg_ast)
            exec(compile(cfg_ast, "config.py", "exec"), module.__dict__)

    return module


def test_norm_domain_key_sanitizes_reserved_chars():
    mod = load_helpers()
    assert mod._norm_domain_key("  a:b?c|d  ") == "a_b_c_d"
    assert mod._norm_domain_key("") == "未分类"
    assert mod._norm_domain_key(None) == "未分类"


def test_build_term_hint_limits_and_formats():
    mod = load_helpers()
    term_dict = {
        "contract": {"target": "合同", "pos": "NOUN", "usage_note": "法律语境"},
        "contract": {"target": "合同", "pos": "NOUN"},  # 去重
        "agreement": "协议",
    }
    hint = mod.build_term_hint(term_dict, "英→中", max_terms=1)

    assert hint.startswith("GLOSSARY (STRICT):\n")
    lines = [line for line in hint.splitlines() if line.startswith("-")]
    assert len(lines) == 1
    assert "contract" in lines[0]
    assert "合同" in lines[0]
    assert "NOUN" in lines[0]


def test_ds_translate_uses_strict_glossary(monkeypatch):
    extra_funcs = {"build_instruction", "ds_translate"}
    mod = load_helpers(extra_funcs)

    captured = {}

    class DummyResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "OK"}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return DummyResp()

    fake_requests = types.SimpleNamespace(post=fake_post)
    mod.requests = fake_requests
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    mod.time = __import__("time")
    mod.log_event = lambda *args, **kwargs: None

    term_dict = {"Contract": {"target": "合同", "pos": "NOUN"}}
    result = mod.ds_translate("text", term_dict, "英→中", "ak", "deepseek-chat")

    assert result == "OK"
    user_msg = captured["payload"]["messages"][-1]["content"]
    assert user_msg.startswith("GLOSSARY (STRICT):\n")
    assert "Contract" in user_msg
    assert "合同" in user_msg
    assert "NOUN" in user_msg
    assert "INSTRUCTION:" in user_msg
    assert "English to Chinese" in user_msg  # 确保方向检测为英译中


def test_build_instruction_detects_direction_variants():
    mod = load_helpers()

    zh_to_en = mod.build_instruction("zh->en")
    en_to_zh = mod.build_instruction("en->zh")

    assert "Chinese to English" in zh_to_en
    assert "English to Chinese" in en_to_zh
