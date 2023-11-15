import pandas as pd
import numpy as np
import datetime as dt
import math
import random
from plotly.figure_factory import create_dendrogram
from itertools import cycle
import matplotlib.pyplot as plt


class group:

    def __init__(self) -> None:
        self.subset_count = 0
        self.sub_set = []
        self.centroid = np.zeros(1)
        self.distance = 0

    def __str__(self) -> str:
        r = self.print_recursive(0)
        return f'{r}'

    def print_recursive(self, level: int) -> str:
        if len(self.sub_set) == 1:
            return str(self.sub_set[0])
        else:
            ret = ''
            for s in self.sub_set:
                ret += "\n" + " " * level
                ret += s.print_recursive(level + 1)
            return f'({self.get_amout_class_rec({})}{ret}' + '\n' + ' ' * level + ")"

    def get_amout_class_rec(self, set_dict: dict):
        if len(self.sub_set) == 1:
            if self.sub_set[0][0] in set_dict:
                set_dict[self.sub_set[0][0]] += 1
            else:
                set_dict[self.sub_set[0][0]] = 1
        else:
            for s in self.sub_set:
                s.get_amout_class_rec(set_dict)
        return set_dict

    def get_as_np_array(self):
        return self._get_groups_at(0)

    def _get_groups_at(self, level: int):
        if len(self.sub_set) == 1:
            return self.sub_set[0][0]
        else:
            ret = np.array([self.sub_set[0]._get_groups_at(level + 1), self.sub_set[1]._get_groups_at(level + 1)])
            return ret

    def _get_all_children_as_groups(self, child_list: list):
        if len(self.sub_set) == 1:
            child_list.append(self.sub_set[0])
        else:
            for s in self.sub_set:
                s._get_all_children_as_groups(child_list)
        return child_list

    def get_groups_at_dist(self, dist: float, groups: list):
        if self.distance > dist:
            for s in self.sub_set:
                s.get_groups_at_dist(dist, groups)
        else:
            if len(self.sub_set) == 1:
                groups.append([self.sub_set[0]])
                return
            children = []
            for s in self.sub_set:
                s._get_all_children_as_groups(children)
            groups.append(children)


def normalize(max, min, value):
    return (value - min) / (max - min)


def diantce_to(x, y):
    _sum = 0
    for i in range(len(x)):
        if not math.isnan(x[i]) and not math.isnan(y[i]):
            _sum += (x[i] - y[i]) ** 2
    return math.sqrt(_sum)


def build_tree(data_vector, data_class):
    set_to_index = []

    for i in range(len(data_class)):
        g = group()
        g.centroid = data_vector[i]
        g.sub_set.append(data_class[i])
        g.subset_count = 1
        set_to_index.append(g)

    # distance matrix
    dist = np.zeros((len(data_vector), len(data_vector)))
    for i in range(len(data_vector)):
        for j in range(i, len(data_vector)):
            dist[i][j] = diantce_to(data_vector[i], data_vector[j])

    while (len(set_to_index) > 1):
        # find shortest distance in distance matrix
        min = dist[0][1]
        min_i = 0
        min_j = 1
        for i in range(len(dist)):
            for j in range(i + 1, len(dist)):
                if dist[i][j] < min:
                    min = dist[i][j]
                    min_i = i
                    min_j = j
        min_set_a = set_to_index[min_i]
        min_set_b = set_to_index[min_j]
        new_set = group()

        new_set.sub_set = [min_set_a, min_set_b]
        new_set.subset_count = min_set_a.subset_count + min_set_b.subset_count
        new_set.centroid = min_set_a.centroid * (min_set_a.subset_count / new_set.subset_count) + min_set_b.centroid * (
                min_set_b.subset_count / new_set.subset_count)
        new_set.distance = min

        larger_index = min_i if min_i > min_j else min_j
        smaller_index = min_i if min_i < min_j else min_j

        set_to_index.pop(larger_index)

        # remove row and colum of the last set
        dist = np.delete(dist, min_j, 0)
        dist = np.delete(dist, min_j, 1)

        # replace the row and colum of the first set with the new set
        # horizontally
        for i in range(smaller_index + 1, len(dist)):
            dist[smaller_index][i] = diantce_to(new_set.centroid, set_to_index[i].centroid)

        # vertically
        for i in range(smaller_index):
            dist[i][smaller_index] = diantce_to(new_set.centroid, set_to_index[i].centroid)

        # add new set to the list
        set_to_index[smaller_index] = new_set
        # print(f"removed index: {larger_index} and merged with {smaller_index} with distance {min}")

    return set_to_index[0]


def main():
    df = pd.read_csv('movie_data.csv', sep=';', header=0)
    df = df.head(100)
    df.drop('imdb_id', axis=1, inplace=True)

    # df['original_title_length'] = df['original_title'].apply(lambda x: len(str(x).split()))
    # df['overview'] = df['overview'].apply(lambda x: len(str(x).split()))
    df.dropna(subset=['release_date'], inplace=True)
    df['release_date'] = df['release_date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    df['release_date'] = df['release_date'].apply(lambda x: x.timestamp())
    # df = df[df['genres'].isin(['Comedy', 'Action', 'Drama'])]
    df = df.drop_duplicates()

    # Normalize every column except 'genres'
    for col in df.columns:
        print(col)
        if col not in ['genres', 'original_title', 'overview']:
            df[col] = df[col].apply(lambda x: normalize(df[col].max(), df[col].min(), x))
    print(df)

    df1 = df[['genres', 'original_title']]

    df2 = df.drop('genres', axis=1)
    df2 = df2.drop('original_title', axis=1)
    df2 = df2.drop('overview', axis=1)

    df1_np = df1.to_numpy()
    df2_np = df2.to_numpy()
    root = build_tree(df2_np, df1_np)
    while True:
        try:
            user_in = float(input("introducza valor entre 0 y 1: "))
        except ValueError:
            print("PONE UN FLOAT CHE NO ES TAN DIFICIL")
            continue
        if user_in < 0 or user_in > 1:
            print("ENTRE 0 y 1 DIJE PQ NO ME HACÉS CASO?")
            continue

        groups = []
        root.get_groups_at_dist(user_in, groups)
        print(f"------ {user_in} ------")
        for g in groups:
            print("-----")
            print(g)
        print("------ ------ ------")


def test_2d():
    amount = 100
    np.random.seed(0)
    point_cloud = np.random.rand(amount, 2)
    print(point_cloud)
    root = build_tree(point_cloud, point_cloud)
    groups = []
    root.get_groups_at_dist(0.52, groups)

    colors = cycle('bgrcmyk')
    for g in groups:
        color = next(colors)
        for point in g:
            plt.scatter(point[0], point[1], color=color, s=1)
    plt.show()


if __name__ == '__main__':
    main()
    # test_2d()

    # X = np.random.rand(15, 1) # 15 samples, with 12 dimensions each
    # print(X)
    # fig = create_dendrogram(X)
    # fig.show()
