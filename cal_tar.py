import os
import pandas as pd
from docx import Document
import matplotlib.pyplot as plt

# =================配置区域=================
# 文件名配置
FILES = {
    "QiYi (Ours)": "3.test_译文.txt",   # 您的系统输出
    "GPT-4": "3_gpt.docx",              # 对照组1
    "DeepSeek": "3_deepseek.docx"       # 对照组2
}

TERM_FILE = "terms.csv"  # 术语表文件名
# =========================================

def read_text_from_file(file_path):
    """根据后缀名读取不同格式的文件内容"""
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到文件 {file_path}，跳过该文件。")
        return ""
    
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif ext == '.docx':
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            print(f"不支持的格式: {ext}")
            return ""
    except Exception as e:
        print(f"读取 {file_path} 时出错: {e}")
        return ""

def load_terms(csv_path):
    """读取术语表，返回目标术语列表"""
    try:
        # 尝试使用 utf-8 读取，如果失败尝试 gbk (适应不同系统的 Excel 保存格式)
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gbk')
            
        # 检查列名
        if 'tgt' not in df.columns:
            raise ValueError("CSV文件中缺少 'tgt' 列 (目标术语列)")
        
        # 去除空值并获取目标术语列表
        terms = df['tgt'].dropna().astype(str).tolist()
        # 去除首尾空格
        terms = [t.strip() for t in terms if t.strip()]
        return terms
    except Exception as e:
        print(f"读取术语表出错: {e}")
        return []

def calculate_tar(text, terms):
    """
    计算术语准确率 (TAR)
    逻辑：检查术语表中的每个目标术语是否出现在译文中
    """
    if not text or not terms:
        return 0.0, [], []

    total_terms = len(terms)
    hit_count = 0
    missing_terms = []
    
    # 简单的字符串包含匹配 (可根据需要改为正则全词匹配)
    for term in terms:
        if term in text:
            hit_count += 1
        else:
            missing_terms.append(term)
            
    tar = (hit_count / total_terms) * 100
    return tar, missing_terms, total_terms

def main():
    print("正在初始化评测脚本...\n")
    
    # 1. 加载术语
    target_terms = load_terms(TERM_FILE)
    if not target_terms:
        print("❌ 术语表为空或读取失败，程序终止。")
        return
    
    print(f"✅ 成功加载术语表，共包含 {len(target_terms)} 个关键术语。\n")
    
    results = {}
    
    # 2. 遍历文件进行计算
    print(f"{'系统名称':<15} | {'TAR (%)':<10} | {'命中/总数':<10}")
    print("-" * 45)
    
    for system_name, file_path in FILES.items():
        # 读取文本
        text = read_text_from_file(file_path)
        
        # 计算 TAR
        tar_score, missed, total = calculate_tar(text, target_terms)
        
        # 存储结果
        results[system_name] = tar_score
        
        # 打印表格行
        print(f"{system_name:<15} | {tar_score:<10.2f} | {total - len(missed)}/{total}")
        
        # (可选) 打印每个文件缺失的前3个术语，方便调试
        # if missed:
        #     print(f"   (缺失示例: {', '.join(missed[:3])}...)")

    print("-" * 45)
    
    # 3. 绘制柱状图 (用于论文)
    if results:
        plt.figure(figsize=(10, 6))
        
        # 定义颜色 (Highlight QiYi)
        colors = ['#4CAF50' if 'QiYi' in name else '#B0BEC5' for name in results.keys()]
        
        bars = plt.bar(results.keys(), results.values(), color=colors, width=0.5)
        
        plt.title('Terminology Accuracy Rate (TAR) Comparison', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, 110) # Y轴留出空间写数字
        
        # 在柱子上标数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, height + 1, 
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        
        # 保存图片
        output_img = 'TAR_Comparison_Result.png'
        plt.savefig(output_img, dpi=300)
        print(f"\n📊 统计图表已生成并保存为: {output_img}")

if __name__ == "__main__":
    main()