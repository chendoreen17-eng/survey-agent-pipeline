import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
import networkx as nx
from scipy.sparse import csr_matrix

# --- 配置区 ---
EMBEDDINGS_FILE = "paper_embeddings.npy"      # 之前生成的1536维向量
METADATA_FILE = "papers_with_vectors.csv"    # 包含 paper_id 的元数据
EDGES_FILE = "citation_edges.csv"            # 引用关系边表
OUTPUT_FILE = "final_structured_domains.csv"
NUM_CLUSTERS = 30 
GRAPH_WEIGHT = 0.4  # 结构信息的权重（0.1~0.5），越大越看重“圈子”，越小越看重“用词”

def run_enhanced_pipeline():
    # 1. 加载语义向量
    print("加载语义向量...")
    X_semantic = np.load(EMBEDDINGS_FILE)
    df = pd.read_csv(METADATA_FILE)
    
    # 2. 提取图结构特征 (Structural Embeddings)
    print("正在计算图结构特征 (SVD on Adjacency Matrix)...")
    # 创建一个 paper_id 到 矩阵索引的映射
    id_map = {pid: i for i, pid in enumerate(df['paper_id'])}
    
    # 构建稀疏邻接矩阵
    edges = pd.read_csv(EDGES_FILE)
    rows = []
    cols = []
    data = []
    for _, edge in edges.iterrows():
        if edge['source_paper_id'] in id_map and edge['target_paper_id'] in id_map:
            rows.append(id_map[edge['source_paper_id']])
            cols.append(id_map[edge['target_paper_id']])
            data.append(1.0) # 引用关系的边权重
            
    adj_matrix = csr_matrix((data, (rows, cols)), shape=(len(df), len(df)))
    
    # 使用奇异值分解 (SVD) 将图结构降维到 64 维
    # 这一步能捕捉到论文在引用网络中的“社交位置”
    svd = TruncatedSVD(n_components=64, random_state=42)
    X_structural = svd.fit_transform(adj_matrix)
    
    # 3. 向量对齐与融合
    print("融合语义与结构特征...")
    # 分别标准化，确保量级一致
    X_semantic_norm = normalize(X_semantic)
    X_structural_norm = normalize(X_structural)
    
    # 融合公式：Combined = Semantic + Weight * Structural
    # 我们将结构特征拼接到语义特征后面
    X_combined = np.hstack((
        X_semantic_norm * (1 - GRAPH_WEIGHT), 
        X_structural_norm * GRAPH_WEIGHT
    ))
    
    # 4. 执行最终聚类
    print(f"执行 K-Means 聚类 (K={NUM_CLUSTERS})...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df['final_domain_id'] = kmeans.fit_predict(X_combined)
    
    # 5. 保存结果
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"聚类完成！结果保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_enhanced_pipeline()