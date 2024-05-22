import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import seaborn as sns
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score, precision_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import cdist
# Chargement des données d'entraînement
data = np.loadtxt('optdigits.tra', delimiter=',', dtype=int)
X_tra = data[:, :-1]
Y_tra = data[:, -1] # labels

# Chargement des données de tests
data_tests = np.loadtxt('optdigits.tes', delimiter=',', dtype=int)
X_tes = data_tests[:, :-1]
Y_tes = data_tests[:, -1] # labels


# On affiche dix chiffres de la BA.
f, axarr = plt.subplots(2,5)
for i in range(2):
    for j in range(5):
        axarr[i,j].imshow(X_tra[i+j].reshape(8, 8), cmap='gray')
        axarr[i,j].set_title(f'Chiffre: {int(Y_tra[i+j])}')
        axarr[i,j].axis("off")



kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(X_tra) # Apprentissage


def to_figure(nb):
    plt.clf()
    plt.figure(figsize=(12, 8))

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.hist(Y_tra[kmeans.labels_ == i], bins=range(11), edgecolor="black", color="red")
        plt.title(f"Cluster {i}")
        plt.xticks(range(10))
    
    plt.tight_layout()
    plt.savefig(f'clusters_desesmorts{nb}.png')

# =================================================== silhouette ======================
opening_naruto = silhouette_score(X_tra, kmeans.labels_)
print(opening_naruto)

list_opening_naruto = []
list_of_x = []
for i in range (10,21):
    list_opening_naruto.append(silhouette_score(X_tra,KMeans(n_clusters=i,random_state=42).fit(X_tra).labels_))
    list_of_x.append(i)
plt.clf()
print(list_opening_naruto)
plt.plot(list_of_x,list_opening_naruto)
plt.savefig("diff_K_clusters.png")

# =============== New Clustering on Train ==========================

new_kmeans = KMeans(n_clusters=13, random_state=42)
new_kmeans.fit(X_tra)

clusters = new_kmeans.labels_
cluster_majority_label = np.zeros(13)

for i in range(13):
    labels, counts = np.unique(Y_tra[clusters == i], return_counts=True)
    cluster_majority_label[i] = labels[np.argmax(counts)]

print(cluster_majority_label)



test_clusters = new_kmeans.predict(X_tes)
y_pred = cluster_majority_label[test_clusters]

conf_matrix = confusion_matrix(Y_tes, y_pred)
accuracy = accuracy_score(Y_tes, y_pred)
precision = precision_score(Y_tes, y_pred, average='macro')

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('conf_matrix_1.png')



# -------------------------------------------- Performance globale --------------------------------------------
print(f'Overall accuracy: {accuracy:.4f}')
print(f'Overall precision: {precision:.4f}')


# --------------------------------------- Analyse (matrice de confusion) ---------------------------------------
print('Confusion matrix:')
print(conf_matrix)






# ==================== dendrogramme ============================

Z = linkage(X_tra, method='ward')

plt.figure(figsize=(25, 10))
dendrogram(Z, truncate_mode='level', p=30)
plt.title('Dendrogramme du Clustering Hiérarchique')
plt.xlabel('Échantillons')
plt.ylabel('Distance')
plt.savefig('dendrogram.png')


# ----------------------- Coupure du dendrogramme -----------------------
k = 10
clusters = fcluster(Z, k, criterion='maxclust')

# Histogrammes par cluster
cluster_labels = {i: [] for i in range(1, k+1)}
for label, cluster in zip(Y_tra, clusters):
    cluster_labels[cluster].append(label)


# Visualisation
fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()
for i in range(1, k+1):
    counts = Counter(cluster_labels[i])
    if counts:
        labels, values = zip(*counts.items())
    else:
        labels, values = [], []
    axes[i-1].bar(labels, values, edgecolor="black")
    axes[i-1].set_title(f'Cluster {i}')
    axes[i-1].set_xlabel('Digit')
    axes[i-1].set_ylabel('Count')
    axes[i-1].set_xlim([0, 9])
    axes[i-1].xaxis.set_ticks(range(10))

plt.tight_layout()
plt.savefig('clusters_2.png')

# ----------------------- Indice de la Silouette -----------------------
silhouette_avg = silhouette_score(X_tra, clusters)

# indice de la Silhouette pour chaque K
silhouette_scores = []
k_values = range(10, 21)

for k in k_values:
    clusters = fcluster(Z, k, criterion='maxclust')
    score = silhouette_score(X_tra, clusters)
    silhouette_scores.append(score)

# Visualisation des scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.savefig('diff_silhouette2.png')


# Comparaison avec K-means
kmeans_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters_kmeans = kmeans.fit_predict(X_tra)
    score_kmeans = silhouette_score(X_tra, clusters_kmeans)
    kmeans_scores.append(score_kmeans)
    print(f"K-means with K={k}, Silhouette Score: {score_kmeans}")

# Visualisation de la comparaison des scores de la silhouette
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o', label='Hierarchical Clustering')
plt.plot(k_values, kmeans_scores, marker='o', linestyle='--', label='K-means',color="red")
plt.title('Silhouette Score Comparison: Hierarchical Clustering vs. K-means')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_scores_2.png')



k = 11
clusters_Dendro = fcluster(Z, k, criterion='maxclust')

cluster_labels = {}
for cluster in np.unique(clusters_Dendro):
    labels = Y_tra[cluster == clusters_Dendro]
    most_common_label = Counter(labels).most_common(1)[0][0]
    cluster_labels[cluster] = most_common_label

    # Classification de la base de test
clusters_test = cdist(X_tes, X_tra, 'euclidean')
closest_clusters = np.argmin(clusters_test, axis=1)
y_pred_Dendro = np.array([cluster_labels[clusters_Dendro[i]]
                             for i in closest_clusters])


conf_matrix = confusion_matrix(Y_tes, y_pred_Dendro)
accuracy = accuracy_score(Y_tes, y_pred_Dendro)
precision = precision_score(Y_tes, y_pred_Dendro, average='macro')

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('conf_matrix_2.png')


# -------------------------------------------- Performance globale --------------------------------------------
print(f'Overall accuracy: {accuracy:.4f}')
print(f'Overall precision: {precision:.4f}')


# --------------------------------------- Analyse (matrice de confusion) ---------------------------------------
print('Confusion matrix:')
print(conf_matrix)