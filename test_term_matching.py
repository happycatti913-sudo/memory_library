# -*- coding: utf-8 -*-
"""
术语匹配测试脚本

用于测试改进后的术语匹配函数是否能正确处理各种情况
"""

import re

def _normalize_term(term: str) -> str:
    """
    规范化术语用于匹配：
    1. 去除前后空格
    2. 转小写
    3. 移除多余空格和特殊字符（包括中文空格）
    """
    if not term:
        return ""
    t = str(term).strip().lower()
    # 移除多余空格（包括中英文空格、制表符等）
    t = re.sub(r'[\s\u3000\u00A0]+', ' ', t)
    # 对于中文术语，也可以考虑移除空格
    if all(ord(c) >= 0x4E00 and ord(c) <= 0x9FFF or c == ' ' for c in t.replace(' ', '')):
        # 纯中文术语，移除所有空格
        t = t.replace(' ', '')
    return t

def _find_term_in_map(src_term: str, term_map: dict[str, str]) -> str | None:
    """
    从术语图中查找源术语的目标术语，支持多种匹配策略。
    """
    if not src_term or not term_map:
        return None
    
    src_term = src_term.strip()
    if not src_term:
        return None
    
    # 策略1：精确匹配
    if src_term in term_map:
        return term_map[src_term]
    
    # 策略2：不区分大小写精确匹配
    src_lower = src_term.lower()
    for k, v in term_map.items():
        if k.lower() == src_lower:
            return v
    
    # 策略3：规范化后匹配
    src_norm = _normalize_term(src_term)
    if not src_norm:
        return None
    
    for k, v in term_map.items():
        if _normalize_term(k) == src_norm:
            return v
    
    # 策略4：子串匹配（对于复合词）
    for k, v in term_map.items():
        if src_norm in _normalize_term(k) or _normalize_term(k) in src_norm:
            # 优先选择更短的匹配（更精确）
            if len(k) < len(src_term) * 1.5 and len(src_term) < len(k) * 1.5:
                return v
    
    return None


def test_term_matching():
    """测试术语匹配"""
    
    # 测试用例
    test_cases = [
        {
            "term_map": {"糖尿病": "diabetes", "高血压": "hypertension"},
            "src_terms": [
                ("糖尿病", "diabetes"),
                ("糖尿病 ", "diabetes"),
                (" 糖尿病", "diabetes"),
                ("DIABETES", None),  # 英文不匹配
                ("高血压", "hypertension"),
            ],
            "description": "基本中文术语匹配"
        },
        {
            "term_map": {"Type 2 Diabetes": "T2DM", "Blood Pressure": "BP"},
            "src_terms": [
                ("Type 2 Diabetes", "T2DM"),
                ("type 2 diabetes", "T2DM"),
                ("TYPE 2 DIABETES", "T2DM"),
                ("Type  2  Diabetes", "T2DM"),  # 多余空格
                ("Blood Pressure", "BP"),
            ],
            "description": "英文术语匹配（不区分大小写）"
        },
        {
            "term_map": {"心肌梗死": "MI", "脑卒中": "stroke"},
            "src_terms": [
                ("心肌梗死", "MI"),
                ("心  肌  梗  死", "MI"),  # 多余空格
                ("脑卒中", "stroke"),
            ],
            "description": "中文复杂术语"
        },
    ]
    
    print("=" * 80)
    print("术语匹配测试")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    
    for test_case in test_cases:
        print(f"\n测试场景: {test_case['description']}")
        print(f"术语库: {test_case['term_map']}")
        print("-" * 80)
        
        term_map = test_case["term_map"]
        for src_term, expected in test_case["src_terms"]:
            result = _find_term_in_map(src_term, term_map)
            status = "✓ PASS" if result == expected else "✗ FAIL"
            
            print(f"{status} | 输入: '{src_term}' | 期望: {expected} | 结果: {result}")
            
            total_tests += 1
            if result == expected:
                passed_tests += 1
    
    print("\n" + "=" * 80)
    print(f"测试结果: {passed_tests}/{total_tests} 通过")
    print("=" * 80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = test_term_matching()
    exit(0 if success else 1)
