import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

class KMeansClustering:
    def __init__(self, data, k, max_iter=300):
        self.data = data
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.clusters = None

    def initialize_centroids_plusplus(self):
        centroids = [self.data[np.random.randint(self.data.shape[0])]]
        for _ in range(1, self.k):
            dist_sq = np.array([min([np.linalg.norm(c-x)**2 for c in centroids]) for x in self.data])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()
            i = np.searchsorted(cumulative_probs, r)
            centroids.append(self.data[i])
        self.centroids = np.array(centroids)

    def k_means_clustering(self):
        self.initialize_centroids_plusplus()
        for iteration in range(self.max_iter):
            clusters = {i: [] for i in range(self.k)}
            for point in self.data:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                cluster_index = np.argmin(distances)
                clusters[cluster_index].append(point)
            new_centroids = []
            for i in range(self.k):
                if len(clusters[i]) > 0:
                    new_centroids.append(np.mean(clusters[i], axis=0))
                else:
                    new_centroids.append(self.data[np.random.randint(len(self.data))])
            new_centroids = np.array(new_centroids)
            if iteration > 0 and np.allclose(self.centroids, new_centroids, rtol=1e-6):
                break
            self.centroids = new_centroids
        self.clusters = clusters

    @staticmethod
    def remove_outliers(data, threshold=1.5):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)
        return data[mask], mask

    def calculate_wcss(self):
        wcss = []
        for k in range(1, self.k + 1):
            self.k = k
            self.k_means_clustering()
            wcss_sum = 0
            for cluster_index in self.clusters:
                points = np.array(self.clusters[cluster_index])
                if points.any():
                    centroid = self.centroids[cluster_index]
                    wcss_sum += np.sum((points - centroid) ** 2)
            wcss.append(wcss_sum)
        return wcss

    def plot_elbow_method(self, max_k):
        wcss = self.calculate_wcss()
        plt.plot(range(1, max_k + 1), wcss, 'bx-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    def plot_clusters(self, feature_indices=(5, 2, 4)):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'purple', 'brown']
        for cluster_index in self.clusters:
            cluster_data = np.array(self.clusters[cluster_index])
            ax.scatter(cluster_data[:, feature_indices[0]], cluster_data[:, feature_indices[1]], cluster_data[:, feature_indices[2]], s=50, c=colors[cluster_index], label=f'Cluster {cluster_index}')
        ax.scatter(self.centroids[:, feature_indices[0]], self.centroids[:, feature_indices[1]], self.centroids[:, feature_indices[2]], s=200, c='black', marker='*', label='Centroids')
        ax.set_title('3D K-Means Clustering')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_zlabel('Feature 3')
        plt.legend()
        plt.show()

    def predict_cluster(self, new_data):
        
        distances = [np.linalg.norm(new_data - centroid) for centroid in self.centroids]
        return np.argmin(distances)


# Utilisation de la classe
data = pd.read_csv('movie_data.csv', sep=';', decimal='.')
data_numerical = data[['budget', 'popularity', 'production_companies', 'production_countries', 'revenue', 'runtime', 'spoken_languages', 'vote_average', 'vote_count']].dropna()

# Suppression des outliers
data_numerical, mask = KMeansClustering.remove_outliers(data_numerical)

# Obtenez les indices des lignes conservées
indices = mask[mask].index

# Synchronisez 'data' avec ces indices
data = data.loc[indices].reset_index(drop=True)

# Poursuivez avec la standardisation et le clustering
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numerical)

# KMeans Clustering
kmeans = KMeansClustering(data_scaled, k=5)
kmeans.k_means_clustering()

# Assignez les clusters à 'data'
data['cluster'] = [kmeans.predict_cluster(row) for row in data_scaled]


# Affectation des clusters à 'data'
if len(data_scaled) == len(data):
    data['cluster'] = [kmeans.predict_cluster(row) for row in data_scaled]
else:
    print("Erreur : les longueurs des DataFrames ne correspondent pas.")

# Trouver le genre dominant dans chaque cluster
genre_dominant_par_cluster = {}
for cluster in range(kmeans.k):
    films_dans_cluster = data[data['cluster'] == cluster]
    comptage_genres = films_dans_cluster['genres'].value_counts()
    genre_dominant = comptage_genres.idxmax() if not comptage_genres.empty else 'Aucun genre dominant'
    genre_dominant_par_cluster[cluster] = genre_dominant

# Affichage des genres dominants
for cluster, genre in genre_dominant_par_cluster.items():
    print(f"Cluster {cluster}: Genre dominant = {genre}")

genres_cibles = ['Action', 'Drama', 'Comedy']

# Créer un dictionnaire pour stocker les résultats
pourcentages_par_cluster = {cluster: {genre: 0 for genre in genres_cibles} for cluster in range(kmeans.k)}

# Calculer les pourcentages
for cluster in range(kmeans.k):
    # Sélectionner les films dans le cluster courant
    films_dans_cluster = data[data['cluster'] == cluster]

    # Calculer le total des films dans le cluster
    total_films = len(films_dans_cluster)

    # Calculer le pourcentage pour chaque genre cible
    if total_films > 0:
        for genre in genres_cibles:
            nombre = films_dans_cluster['genres'].str.contains(genre).sum()
            pourcentage = (nombre / total_films) * 100
            pourcentages_par_cluster[cluster][genre] = pourcentage

# Afficher les résultats
for cluster, genres_info in pourcentages_par_cluster.items():
    print(f"Cluster {cluster}:")
    for genre, pourcentage in genres_info.items():
        print(f"  {genre}: {pourcentage:.2f}%")
    print()