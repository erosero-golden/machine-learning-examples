import numpy as np
from sklearn.cluster import KMeans

def divisive_clustering(X, max_clusters=4, random_state=42):
    print(f"Divisive clustering with max_clusters={max_clusters}")
    clusters = [np.arange(len(X))]  # Índices de filas iniciales

    while len(clusters) < max_clusters:
        sizes = [len(c) for c in clusters]
        idx_to_split = np.argmax(sizes)
        indices = np.array(clusters.pop(idx_to_split))  # aseguramos array NumPy

        kmeans = KMeans(n_clusters=2, random_state=random_state)
        labels = kmeans.fit_predict(X.iloc[indices])  # ✅ uso correcto de iloc

        new_cluster1 = indices[labels == 0]
        new_cluster2 = indices[labels == 1]

        clusters.append(new_cluster1)
        clusters.append(new_cluster2)

    final_labels = np.zeros(len(X), dtype=int)
    for cluster_idx, sample_indices in enumerate(clusters):
        final_labels[sample_indices] = cluster_idx

    return final_labels