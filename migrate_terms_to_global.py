# -*- coding: utf-8 -*-
"""
把项目 #1 的术语改为全局术语
"""

import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kb.db")

print("=" * 80)
print("把项目 #1 的术语改为全局术语")
print("=" * 80)

try:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # 检查当前状态
    print("\n【当前状态】")
    stats = cur.execute("""
        SELECT 
            CASE WHEN project_id IS NULL THEN '全局术语库' ELSE '项目 #' || project_id END as category,
            COUNT(*) as cnt
        FROM term_ext
        GROUP BY project_id
        ORDER BY project_id
    """).fetchall()
    
    for category, cnt in stats:
        print(f"  {category}: {cnt} 条")
    
    # 询问确认
    response = input("\n是否继续执行转换? (输入 'y' 确认，其他任何输入都会取消): ").strip().lower()
    if response != 'y':
        print("操作已取消")
        conn.close()
        exit(0)
    
    # 执行转换
    print("\n【执行转换】")
    print("正在把项目 #1 的术语改为全局术语...")
    
    cur.execute("UPDATE term_ext SET project_id = NULL WHERE project_id = 1")
    affected = cur.rowcount
    
    print(f"✓ 已更新 {affected} 条记录")
    
    conn.commit()
    
    # 确认结果
    print("\n【转换后状态】")
    stats = cur.execute("""
        SELECT 
            CASE WHEN project_id IS NULL THEN '全局术语库' ELSE '项目 #' || project_id END as category,
            COUNT(*) as cnt
        FROM term_ext
        GROUP BY project_id
        ORDER BY project_id
    """).fetchall()
    
    for category, cnt in stats:
        print(f"  {category}: {cnt} 条")
    
    # 查看转换后的领域分布
    print("\n【转换后领域分布】")
    domains = cur.execute("""
        SELECT domain, COUNT(*) as cnt
        FROM term_ext
        WHERE domain IS NOT NULL AND domain != ''
        GROUP BY domain
        ORDER BY cnt DESC
    """).fetchall()
    
    for domain, cnt in domains:
        print(f"  • {domain}: {cnt} 条")
    
    conn.close()
    
    print("\n" + "=" * 80)
    print("✓ 转换完成！")
    print("  现在所有项目都可以访问全部 2090 条术语")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
