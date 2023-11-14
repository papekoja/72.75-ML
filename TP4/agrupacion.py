import pandas as pd
import numpy as np
import datetime as dt
import math
import random

class group:
    
    def __init__(self) -> None:
        self.subset_count = 0
        self.sub_set = []
        self.centroid = np.zeros(1)

    def __str__(self) -> str:
        r = self.print_recursive(0)
        return f'{r}'

    def print_recursive(self, level:int) -> str:
        if len(self.sub_set) == 1:
            return str(self.sub_set[0])
        else:
            ret = ''
            for s in self.sub_set:
                ret+="\n"+" "*level
                ret+=s.print_recursive(level+1)
            return f'({self.get_amout_class_rec({})}{ret}' + '\n' + ' '*level + ")"
        
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

def normalize(max, min, value):
    return (value - min) / (max - min)

def diantce_to(x,y):
    _sum = 0
    for i in range(len(x)):
        if not math.isnan(x[i]) and not math.isnan(y[i]):
            _sum += (x[i]-y[i])**2
    return math.sqrt(_sum)

def build_tree(data_vector, data_class):

    set_to_index = []

    for i in range(len(data_class)):
        g = group()
        g.centroid =  data_vector[i]
        g.sub_set.append(data_class[i])
        g.subset_count = 1
        set_to_index.append(g)

    # distance matrix
    dist = np.zeros((len(data_vector), len(data_vector)))
    for i in range(len(data_vector)):
        for j in range(i,len(data_vector)):
            dist[i][j] = diantce_to(data_vector[i], data_vector[j])

    while(len(set_to_index) > 1):
        #find shortest distance in distance matrix
        min = dist[0][1]
        min_i = 0
        min_j = 1
        for i in range(len(dist)):
            for j in range(i+1,len(dist)):
                if dist[i][j] < min:
                    min = dist[i][j]
                    min_i = i
                    min_j = j
        min_set_a = set_to_index[min_i]
        min_set_b = set_to_index[min_j]
        new_set = group()

        new_set.sub_set = [min_set_a, min_set_b]
        new_set.subset_count = min_set_a.subset_count + min_set_b.subset_count
        new_set.centroid = min_set_a.centroid * (min_set_a.subset_count/new_set.subset_count) + min_set_b.centroid * (min_set_b.subset_count/new_set.subset_count)

        larger_index = min_i if min_i > min_j else min_j
        smaller_index = min_i if min_i < min_j else min_j

        set_to_index.pop(larger_index)

        #remove row and colum of the last set
        dist = np.delete(dist, min_j, 0)
        dist = np.delete(dist, min_j, 1)

        #replace the row and colum of the first set with the new set
        #horizontally
        for i in range(smaller_index + 1, len(dist)):
            dist[smaller_index][i] = diantce_to(new_set.centroid, set_to_index[i].centroid)

        #vertically
        for i in range(smaller_index):
            dist[i][smaller_index] = diantce_to(new_set.centroid, set_to_index[i].centroid)
        
        #add new set to the list
        set_to_index[smaller_index] = new_set
        print(f"removed index: {larger_index} and merged with {smaller_index} with distance {min}")

    return set_to_index[0]


def main():
    df = pd.read_csv('movie_data.csv', sep=';', header=0)
    df = df.head(1000)
    df.drop('imdb_id', axis=1, inplace=True)

    #df['original_title_length'] = df['original_title'].apply(lambda x: len(str(x).split()))
    #df['overview'] = df['overview'].apply(lambda x: len(str(x).split()))
    df.dropna(subset=['release_date'], inplace=True)
    df['release_date'] = df['release_date'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d'))
    df['release_date'] = df['release_date'].apply(lambda x: x.timestamp())
    df = df[df['genres'].isin(['Comedy', 'Action', 'Drama'])]

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
    build_tree(df2_np, df1_np)

def test_2d():
    amount = 200
    point_cloud = np.random.rand(amount, 2)

    print(point_cloud)
    print(build_tree(point_cloud, point_cloud))

if __name__ == '__main__':
    #main()
    test_2d()