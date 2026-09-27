import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

np.random.seed(0)
X, y = make_blobs(
    n_samples=5000,
    centers=[[4, 4], [-2, -1], [2, -3], [1, 1]],
    cluster_std=0.9,
)

k_means = KMeans(init='k-means++', n_clusters=4, n_init=12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_

fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means_cluster_centers)), colors):
    my_members = (k_means_labels == k)
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks([])
ax.set_yticks([])
plt.close(fig)

k_means3 = KMeans(init='k-means++', n_clusters=3, n_init=12)
k_means3.fit(X)
print('three_cluster_count=', len(set(k_means3.labels_)))

cust_df = pd.read_csv('Cust_Segmentation.csv')
df = cust_df.drop('Address', axis=1)
X2 = df.values[:, 1:]
X2 = np.nan_to_num(X2)
clusterNum = 3
k_means = KMeans(init='k-means++', n_clusters=clusterNum, n_init=12)
k_means.fit(X2)
labels = k_means.labels_
df['Clus_km'] = labels
print('customer_cluster_count=', df['Clus_km'].nunique())
print('mean_rows=', df.groupby('Clus_km').mean().shape[0])
print('verification ok')
