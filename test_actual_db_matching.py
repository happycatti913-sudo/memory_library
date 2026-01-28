# -*- coding: utf-8 -*-
"""
测试改进的术语匹配函数
"""

import sqlite3
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kb.db")

def _normalize_term(term: str) -> str:
    """规范化术语"""
    if not term:
        return ""
    t = str(term).strip().lower()
    t = re.sub(r'[\s\u3000\u00A0]+', ' ', t)
    if all(ord(c) >= 0x4E00 and ord(c) <= 0x9FFF or c == ' ' for c in t.replace(' ', '')):
        t = t.replace(' ', '')
    return t

def _find_term_in_map(src_term: str, term_map: dict[str, str]) -> str | None:
    """改进的术语匹配"""
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
            if len(k) < len(src_term) * 1.5 and len(src_term) < len(k) * 1.5:
                return v
    
    return None

def test_matching_on_db():
    """在实际数据库上测试匹配"""
    if not os.path.exists(DB_PATH):
        print(f"❌ 数据库不存在: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    try:
        print("=" * 80)
        print("测试改进的术语匹配函数")
        print("=" * 80)
        
        # 获取项目 #2 的术语
        rows = cur.execute("""
            SELECT source_term, target_term, domain, project_id,
                   CASE WHEN project_id = ? THEN 'static' ELSE 'dynamic' END as origin
            FROM term_ext
            WHERE project_id = ? OR project_id IS NULL
            ORDER BY CASE WHEN project_id = ? THEN 0 ELSE 1 END
        """, (2, 2, 2)).fetchall()
        
        print(f"\n获取项目 #2 的术语: {len(rows)} 条\n")
        
        # 构建术语map
        term_map = {}
        for src, tgt, domain, pid, origin in rows:
            if src:
                src_raw = src.strip()
                tgt_raw = tgt.strip() if tgt else ""
                term_map[src_raw] = tgt_raw
        
        # 测试用例
        test_cases = [
            ("复发性或难治性急性髓系白血病", "精确匹配"),
            ("复发性或难治性急性髓系白血病 ", "尾部有空格"),
            (" 复发性或难治性急性髓系白血病", "首部有空格"),
            ("复发性或难治性急性髓系白血病  ", "多个尾部空格"),
            ("DOSE-ESCALATION", "英文大写"),
            ("Dose-escalation", "英文首字大写"),
            ("dose-escalation", "英文全小写"),
        ]
        
        print("测试结果对比：")
        print("-" * 80)
        print(f"{'测试术语':<40} {'旧方法':<25} {'新方法':<30}")
        print("-" * 80)
        
        for test_term, description in test_cases:
            # 旧方法
            old_result = term_map.get(test_term) or term_map.get(test_term.lower()) or "❌ 无"
            
            # 新方法
            new_result = _find_term_in_map(test_term, term_map)
            new_display = new_result if new_result else "❌ 无"
            
            # 简化显示
            test_display = f"{test_term[:20]}"
            old_display = f"{str(old_result)[:20]}" if old_result != "❌ 无" else "❌"
            new_display = f"{str(new_result)[:20]}" if new_result else "❌"
            
            match_status = "✓" if new_result and not (old_result != "❌ 无") else "→"
            
            print(f"{test_display:<40} {old_display:<25} {match_status} {new_display:<27}")
        
        print("\n" + "=" * 80)
        print("改进总结：")
        print("  ✓ 新方法使用 4 级匹配策略")
        print("  ✓ 能处理空格、大小写等格式差异")
        print("  ✓ 降低术语无法匹配的情况")
        print("=" * 80)
    
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    test_matching_on_db()
