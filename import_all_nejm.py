import os
import sqlite3
from tqdm import tqdm

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "kb.db")

# 定义要处理的文件对
FILE_GROUPS = [
    ("nejm.train.zh", "nejm.train.en", "train_set"),
    ("nejm.dev.zh",   "nejm.dev.en",   "dev_set"),
    ("nejm.test.zh",  "nejm.test.en",  "test_set")
]

def get_category_info(en_text):
    """自动分类逻辑"""
    text = str(en_text).lower()
    if any(k in text for k in ['indication', 'dosage', 'adverse', 'side effect']):
        return "医药说明书", "Drug_Labels"
    if any(k in text for k in ['trial', 'patient', 'randomized', 'conclusion', 'methods']):
        return "临床研究报告", "Clinical_CSR"
    return "通用医疗", "Medical_General"

def run_batch_import():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    total_all = 0

    for zh_name, en_name, set_tag in FILE_GROUPS:
        zh_path = os.path.join(BASE_DIR, zh_name)
        en_path = os.path.join(BASE_DIR, en_name)

        if not os.path.exists(zh_path) or not os.path.exists(en_path):
            print(f"⚠️ 跳过 {set_tag}: 找不到文件 {zh_name} 或 {en_name}")
            continue

        print(f"🚀 正在搬运 {set_tag} 分组数据...")
        insert_rows = []
        batch_size = 5000
        current_set_count = 0

        # 同时读取中英文件
        with open(zh_path, 'r', encoding='utf-8', errors='ignore') as f_zh, \
             open(en_path, 'r', encoding='utf-8', errors='ignore') as f_en:
            
            for zh_line, en_line in tqdm(zip(f_zh, f_en), desc=f"进度-{set_tag}"):
                zh, en = zh_line.strip(), en_line.strip()
                if not zh or not en: continue
                
                dom_cn, title_en = get_category_info(en)
                # 写入系统 corpus 表，note 标记具体来源
                insert_rows.append((
                    f"NEJM_{set_tag}_{title_en}", 
                    999, "英译中", en, zh, 
                    f"nejm_full_import_{set_tag}", dom_cn
                ))
                
                if len(insert_rows) >= batch_size:
                    cur.executemany("INSERT INTO corpus (title, project_id, lang_pair, src_text, tgt_text, note, domain) VALUES (?,?,?,?,?,?,?)", insert_rows)
                    conn.commit()
                    current_set_count += len(insert_rows)
                    insert_rows = []

            if insert_rows:
                cur.executemany("INSERT INTO corpus (title, project_id, lang_pair, src_text, tgt_text, note, domain) VALUES (?,?,?,?,?,?,?)", insert_rows)
                conn.commit()
                current_set_count += len(insert_rows)
        
        print(f"✅ {set_tag} 完成，导入 {current_set_count} 条。")
        total_all += current_set_count

    conn.close()
    print(f"\n🎉 全部任务完成！共计 {total_all} 条语料已存入数据库。")

if __name__ == "__main__":
    run_batch_import()