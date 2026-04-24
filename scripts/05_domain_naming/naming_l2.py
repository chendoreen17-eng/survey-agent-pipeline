import pandas as pd
import requests
import json
import os

# --- 配置 ---
API_KEY = os.getenv("LLMMELON_API_KEY", "").strip()
API_URL = "https://llmmelon.cloud/v1/chat/completions"
MODEL = "claude-3-5-sonnet" # 或 gpt-4o
INPUT_FILE = "final_structured_domains.csv"

def get_llm_response(prompt):
    if not API_KEY:
        raise ValueError("LLMMELON_API_KEY 为空，请先在终端设置环境变量。")
    try:
        API_KEY.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError("LLMMELON_API_KEY 包含非法字符（疑似仍是中文占位符）。") from exc

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data['choices'][0]['message']['content']

def name_l2_domains():
    df = pd.read_csv(INPUT_FILE)
    # 防御式清洗，避免空列/数值列导致 .str 报错
    if 'title' not in df.columns:
        df['title'] = ''
    if 'abstract' not in df.columns:
        df['abstract'] = ''
    if 'citation_count' not in df.columns:
        df['citation_count'] = 0

    df['title'] = df['title'].fillna('').astype(str)
    df['abstract'] = df['abstract'].fillna('').astype(str)
    df['citation_count'] = pd.to_numeric(df['citation_count'], errors='coerce').fillna(0)
    l2_results = []

    for cluster_id in sorted(df['final_domain_id'].unique()):
        # 1. 提取该簇的信息
        cluster_data = df[df['final_domain_id'] == cluster_id]
        
        # 选取被引量最高的 5 篇论文标题
        top_titles = cluster_data.nlargest(5, 'citation_count')['title'].tolist()
        titles_str = "\n- ".join(top_titles)
        
        # 选取 2 条代表性摘要（前300字）
        sample_abstracts = cluster_data.nlargest(2, 'citation_count')['abstract'].str[:300].tolist()
        abstracts_str = "\n".join(sample_abstracts)

        # 2. 构建 Prompt
        prompt = f"""你是一个资深学术分析专家。请根据以下论文聚类的信息，为该研究子方向拟定一个精准的学术名称。

【代表性论文标题】：
- {titles_str}

【核心摘要片段】：
{abstracts_str}

【任务】：
请给出一个 5-10 字的中文学术名称（如：“基于知识蒸馏的模型压缩”），并附带对应的英文名称。
只返回结果，格式如下：
中文名称：xxx
英文名称：xxx
核心研究对象：(一句话描述)"""

        print(f"正在为 Cluster {cluster_id} 命名...")
        res = get_llm_response(prompt)
        l2_results.append({"final_domain_id": cluster_id, "l2_label": res})

    # 保存 L2 命名结果
    l2_df = pd.DataFrame(l2_results)
    l2_df.to_csv("l2_domain_names.csv", index=False)
    print("二级 Domain 命名完成！")

if __name__ == "__main__":
    name_l2_domains()
