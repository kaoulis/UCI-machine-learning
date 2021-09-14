import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
import scipy.cluster as sc

# Load data
def csv_to_nparray(csv):
    dataRaw = []
    DataFile = open(csv, "r")
    while True:
        theLine = DataFile.readline()
        if len(theLine) == 0:
            break
        readData = theLine.split(",")
        for pos in range(len(readData)):
            readData[pos] = float(readData[pos])
        dataRaw.append(readData)
    DataFile.close()
    return np.array(dataRaw)
data_array = csv_to_nparray("HTRU_2.csv")
df = pd.DataFrame(data=data_array)

# Data Preprocessing
# 1. Check for missing values
print("Are there null values?", np.isnan(np.sum(data_array)))

# Imbalanced data
print((df[8][df[8]==1]).count())
print((df[8][df[8]==0]).count())

# Seperate class target
labels = data_array[:, 8]
data_array = np.delete(data_array, 8, axis=1)

# Standardization
def standard(data):
    standard_data = data.copy()
    rows = data.shape[0]
    cols = data.shape[1]

    for j in range(cols):
        sigma = np.std(data[:, j])
        mu = np.mean(data[:, j])
        for i in range(rows):
            standard_data[i, j] = (data[i, j] - mu) / sigma

    return standard_data
standard_data = standard(data_array)
df = pd.DataFrame(data=standard_data)

# PCA
scaled_data = standard(data_array)
pca = PCA()
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)

# Plot scree plot
var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(var) + 1)]
plt.figure(figsize=(4, 4))
plt.bar(x=range(1, len(var) + 1), height=var, tick_label=labels)
plt.ylabel('Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
# plt.savefig("plt_scree.png")
plt.show()

# PLot elbow
plt.figure(figsize=(4, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('Elbow Plot')
# plt.savefig("plt_elbow.png")
plt.show()

# Set the numbers of PCs
pca = PCA(n_components=5)
afterPCA = pca.fit_transform(scaled_data)

#############################
# 5. Clustering #############
#############################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Agglomerative Hierarchical clustering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot dentrogram
plt.figure(figsize=(10, 7))
plt.title("Dendogram")
dend = sc.hierarchy.dendrogram(sc.hierarchy.linkage(afterPCA, method='ward'))
plt.axhline(65)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.xlabel('Observations')
plt.ylabel('Height')
# plt.savefig("plt_dendrogram.png")
plt.show()

# Set 9 clusters
cluster = AgglomerativeClustering(n_clusters=9, affinity='euclidean', linkage='ward')
cluster.fit_predict(afterPCA)

# plot scatter plot for PC0 and PC1 with clusters
plt.figure(figsize=(10, 7))
plt.scatter(afterPCA[:,0], afterPCA[:,1], c=cluster.labels_, cmap='rainbow')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.savefig("plt_HC_clusters.png")
plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ k means ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dislist = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=17898, n_init=10, random_state=0)
    kmeans.fit(afterPCA)
    dislist.append(kmeans.inertia_)

# PLot k means elbow
plt.figure(figsize=(10, 7))
plt.plot(range(1, 11), dislist)
plt.xlabel('Number of clusters')
plt.ylabel('Variation')
# plt.savefig("plt_elbow_clusters.png")
plt.show()

# Fit the algorithm
kmeans = KMeans(n_clusters=9, max_iter=17898, n_init=10, random_state=0)
kmeans.fit_predict(afterPCA)
pred_y = kmeans.fit_predict(afterPCA)

# Plot scatterplot with clusters
plt.figure(figsize=(10, 7))
plt.scatter(afterPCA[:,0], afterPCA[:,1], c=kmeans.labels_, cmap='rainbow')
plt.xlabel('PC1')
plt.ylabel('PC2')
# plt.savefig("plt_kmean_clusters.png")
plt.show()