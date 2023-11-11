import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


def initialize_centroids_plusplus(data, k):
    # Choisir un premier centroïde au hasard parmi les données
    centroids = [data[np.random.randint(data.shape[0])]]
    # Choisir les k-1 centroïdes restants
    for _ in range(1, k):
        dist_sq = np.array([min([np.linalg.norm(c-x)**2 for c in centroids]) for x in data])
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        i = np.searchsorted(cumulative_probs, r)
        centroids.append(data[i])
    return np.array(centroids)

def k_means_clustering(data, k, max_iter):
    centroids = initialize_centroids_plusplus(data, k)
    for iteration in range(max_iter):
        clusters = {i: [] for i in range(k)}
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)
        new_centroids = []
        for i in range(k):
            if len(clusters[i]) > 0:
                new_centroids.append(np.mean(clusters[i], axis=0))
            else:
                new_centroids.append(data[np.random.randint(len(data))])
        centroids = np.array(new_centroids)
        if iteration > 0 and np.allclose(old_centroids, centroids, rtol=1e-6):
            break
        old_centroids = centroids
    return centroids, clusters

def remove_outliers(data, threshold=1.5):
    # Calculez l'écart interquartile (IQR)
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    # Filtrez les outliers
    data_filtered = data[~((data < (Q1 - threshold * IQR)) |(data > (Q3 + threshold * IQR))).any(axis=1)]
    return data_filtered

import numpy as np
import matplotlib.pyplot as plt

def calculate_wcss(data, max_k):
    wcss = []
    for k in range(1, max_k + 1):
        centroids, clusters = k_means_clustering(data, k, 100)
        wcss_sum = 0
        for cluster_index in clusters:
            points = np.array(clusters[cluster_index])
            if points.any():  # If cluster is not empty
                centroid = centroids[cluster_index]
                wcss_sum += np.sum((points - centroid) ** 2)
        wcss.append(wcss_sum)
    return wcss

def plot_elbow_method(data, max_k):
    wcss = calculate_wcss(data, max_k)
    plt.plot(range(1, max_k + 1), wcss, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

# Vous devrez remplacer 'data_scaled' par vos données normalisées réelles.



# Load data and preprocess
data = pd.read_csv('movie_data.csv', sep=';', decimal='.')
data_numerical = data[['budget', 'popularity','production_companies','production_countries','revenue','runtime','spoken_languages','vote_average','vote_count']].dropna()

# Remove outliers (Code for this function is assumed to be correct)
data_numerical = remove_outliers(data_numerical)

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numerical)

#plot_elbow_method(data_scaled, max_k=10)

# Assuming we have determined the optimal k to be 4
optimal_k = 5
centroids, clusters = k_means_clustering(data_scaled, optimal_k, 300)
'''
# Plotting the clusters and centroids in 2D
feature1_index, feature2_index = 0, 1  # Remplacer par les indices appropriés
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k',  'orange', 'purple', 'brown']
for cluster_index in clusters:
    cluster_data = np.array(clusters[cluster_index])
    plt.scatter(cluster_data[:, feature1_index], cluster_data[:, feature2_index], s=50, c=colors[cluster_index], label=f'Cluster {cluster_index}')
centroids_array = np.array(centroids)
plt.scatter(centroids_array[:, 0], centroids_array[:, 1], s=200, c='black', marker='*', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Normalized Budget')
plt.ylabel('Normalized Popularity')
plt.legend()
plt.show()
'''

# Plotting the clusters and centroids in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

feature1_index, feature2_index, feature3_index = 0, 1, 2  # Remplacer par les indices appropriés
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown']
for cluster_index in clusters:
    cluster_data = np.array(clusters[cluster_index])
    ax.scatter(cluster_data[:, feature1_index], cluster_data[:, feature2_index], cluster_data[:, feature3_index], s=50, c=colors[cluster_index], label=f'Cluster {cluster_index}')

centroids_array = np.array(centroids)
ax.scatter(centroids_array[:, feature1_index], centroids_array[:, feature2_index], centroids_array[:, feature3_index], s=200, c='black', marker='*', label='Centroids')

ax.set_title('3D K-Means Clustering')
ax.set_xlabel('Feature 1')  # Remplacer par le nom de la variable
ax.set_ylabel('Feature 2')  # Remplacer par le nom de la variable
ax.set_zlabel('Feature 3')  # Remplacer par le nom de la variable
plt.legend()
plt.show()
