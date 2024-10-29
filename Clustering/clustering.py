import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from minisom import MiniSom

# 加载Iris数据集
iris = datasets.load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# 数据标准化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# K-means聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)

# 自组织映射神经网络（SOM）
som = MiniSom(3, 1, input_len=data_scaled.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(data_scaled)
som.train(data_scaled, 1000)

# 获取每个样本的簇标签
som_labels = np.array([som.winner(x) for x in data_scaled])
som_labels = np.array([0 if x[0] == 0 else 1 if x[0] == 1 else 2 for x in som_labels])

# AP聚类
ap_labels = AgglomerativeClustering(n_clusters=3).fit_predict(data_scaled)

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_scaled)

# 绘图比较聚类结果
plt.figure(figsize=(16, 12))

# K-means
plt.subplot(221)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-means Clustering')

# SOM
plt.subplot(222)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=som_labels, cmap='viridis')
plt.title('Self-Organizing Map Clustering')

# AP聚类
plt.subplot(223)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=ap_labels, cmap='viridis')
plt.title('Affinity Propagation Clustering')

# DBSCAN
plt.subplot(224)
plt.scatter(data_scaled[:, 0], data_scaled[:, 1], c=dbscan.labels_, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.tight_layout()
plt.show()

# 输出轮廓系数
print(f"K-means Silhouette Score: {silhouette_score(data_scaled, kmeans_labels):.2f}")
print(f"SOM Silhouette Score: {silhouette_score(data_scaled, som_labels):.2f}")
print(f"AP Clustering Silhouette Score: {silhouette_score(data_scaled, ap_labels):.2f}")

# 处理DBSCAN的负类（噪声点）
if len(set(dbscan_labels)) > 1:  # 检查是否有有效的簇
    valid_dbscan_labels = dbscan_labels[dbscan_labels >= 0]  # 只保留有效类的标签
    print(f"DBSCAN Silhouette Score: {silhouette_score(data_scaled[dbscan_labels >= 0], valid_dbscan_labels):.2f}")
else:
    print("DBSCAN没有有效的类簇。")
