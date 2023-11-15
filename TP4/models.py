import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D



class SOM:
    def __init__(self, height, width, input_dim):
        self.height = height
        self.width = width
        self.input_dim = input_dim
        self.weights = np.random.random((height, width, input_dim))

    def find_bmu(self, input_vec):
        min_dist = np.inf
        bmu_idx = (0, 0)
        for i in range(self.height):
            for j in range(self.width):
                w = self.weights[i, j, :]
                dist = np.linalg.norm(w - input_vec)
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        return bmu_idx

    def update_weights(self, input_vec, bmu_idx, learning_rate, radius):
        for i in range(self.height):
            for j in range(self.width):
                w = self.weights[i, j, :]
                dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                if dist_to_bmu <= radius:
                    influence = np.exp(-dist_to_bmu / (2 * (radius ** 2)))
                    new_w = w + learning_rate * influence * (input_vec - w)
                    self.weights[i, j, :] = new_w

    def train(self, data, num_epochs, learning_rate, radius):
        for epoch in range(num_epochs):
            for input_vec in data:
                bmu_idx = self.find_bmu(input_vec)
                self.update_weights(input_vec, bmu_idx, learning_rate, radius)

    def calculate_distance_matrix(self):
          distance_matrix = np.zeros((self.height, self.width))
          for i in range(self.height):
              for j in range(self.width):
                  if i > 0:
                      distance_matrix[i, j] += np.linalg.norm(self.weights[i, j, :] - self.weights[i-1, j, :])
                  if i < self.height - 1:
                      distance_matrix[i, j] += np.linalg.norm(self.weights[i, j, :] - self.weights[i+1, j, :])
                  if j > 0:
                      distance_matrix[i, j] += np.linalg.norm(self.weights[i, j, :] - self.weights[i, j-1, :])
                  if j < self.width - 1:
                      distance_matrix[i, j] += np.linalg.norm(self.weights[i, j, :] - self.weights[i, j+1, :])
          return distance_matrix

    def create_u_matrix(self):
          u_matrix = self.calculate_distance_matrix()
          return u_matrix
    



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
        data_filtered = data[~((data < (Q1 - threshold * IQR)) | (data > (Q3 + threshold * IQR))).any(axis=1)]
        return data_filtered

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

    def plot_clusters(self, feature_indices=(0, 1, 2)):
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