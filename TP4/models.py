import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


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