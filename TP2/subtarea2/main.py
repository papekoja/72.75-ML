import math
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt


def data_clean_up():
    _df = pd.read_csv('reviews_sentiment.csv', sep=';')
    unusable = _df[_df['titleSentiment'].isna()].index
    _df.drop(unusable, inplace=True)
    return _df


# Los comentarios valorados con 1 estrella, ¿qué cantidad promedio de palabras tienen?
def task1(_df: pandas.DataFrame):
    one_star_rating = _df[_df["Star Rating"] == 1]

    one_star_avg_words = one_star_rating["wordcount"].sum() / len(one_star_rating)
    print(f'Cantidad promedio de palabras: {one_star_avg_words}')


# Dividir el conjunto de datos en un conjunto de entrenamiento y otro de prueba.

# Stars Raitng y como variables explicativas las
# variables numéricas: wordcount, Title sentiment, sentimentValue

def update_min_max(min_max: list, row):
    if row < min_max[0]:
        min_max[0] = row
    if row > min_max[1]:
        min_max[1] = row


def task2(_df: pandas.DataFrame, split_training) -> tuple:
    word_count_range = [0, 0]
    sentiment_value_range = [0, 0]
    for index, row in _df.iterrows():
        update_min_max(word_count_range, row["wordcount"])
        update_min_max(sentiment_value_range, row["sentimentValue"])

    df_training = []
    df_test = []

    # get smalles class
    min_amount = len(_df)
    for x in range(1, 6):
        l = len(_df[_df["Star Rating"] == x])
        if l < min_amount:
            min_amount = l

    for x in range(1, 6):
        df_temp = _df[_df["Star Rating"] == x]
        # cut down every class to the size of the smalles class
        df_temp = df_temp.iloc[:min_amount, :]

        df_temp = df_temp.replace("negative", 0)
        df_temp = df_temp.replace("positive", 1)

        df_temp = df_temp[["Star Rating", "wordcount", "titleSentiment", "sentimentValue"]].to_numpy()
        # normalize data
        addition = np.array(
            [0,
             word_count_range[0],
             0,
             -sentiment_value_range[0]
             ])

        division = np.array(
            [1,
             1 / (word_count_range[1] - word_count_range[0]),
             1,
             1 / (sentiment_value_range[1] - sentiment_value_range[0])
             ])

        for row in df_temp:
            row += addition
            row *= division

        rows_temp = df_temp.shape[0]

        split = int(rows_temp * split_training)
        df_training_temp = df_temp[:split]
        df_test_temp = df_temp[split:]

        df_training.extend(df_training_temp.tolist())
        df_test.extend(df_test_temp.tolist())

    return df_training, df_test


def distance(x_class_1: list, x_class_2: list):
    sum_distances = 0
    for n in range(1, len(x_class_1)):
        sum_distances += (x_class_1[n] - x_class_2[n]) ** 2
    return math.sqrt(sum_distances)


def k_nn(k: int, class_x_new_instance: list, class_x_instances: list, weighted=False):
    distance_instance = []
    # sort by distance to newly inserted instance
    for x_class in class_x_instances:
        d = distance(class_x_new_instance, x_class)
        distance_instance.append((d, x_class))

    distance_instance.sort(key=lambda tup: tup[0])

    class_amount_in_first_k = {}
    for _k in range(k):
        c = int(distance_instance[_k][1][0])
        if c in class_amount_in_first_k.keys():

            class_amount_in_first_k[c] += 1 if not weighted else 1 / (distance_instance[_k][0]) ** 2
        else:
            class_amount_in_first_k[c] = 1 if not weighted else 1 / (distance_instance[_k][0]) ** 2

    amount = list(class_amount_in_first_k.values())
    amount.sort(reverse=True)

    return_list = []
    for key in class_amount_in_first_k.keys():
        amount_of_key = class_amount_in_first_k[key]
        if amount_of_key == amount[0]:
            return_list.append(key)
    return return_list


def task3(_df_training, _df_test):
    _k = 60
    for test in _df_test:
        print(
            f'{test[0]} -> {k_nn(_k, test, _df_training, weighted=False)},'
            f' weighted: {k_nn(_k, test, _df_training, weighted=True)}')


def task4(_df_training, _df_test):
    labels = range(1, 6)

    _k = 3
    hit_matrix = np.zeros((len(labels), len(labels)))
    true_positives = [0] * 5
    false_positives = [0] * 5
    for test in _df_test:
        result_k_nn = k_nn(_k, test, _df_training, weighted=True)
        if len(result_k_nn) == 1:
            if int(test[0]) == result_k_nn[0]:
                true_positives[int(test[0]) - 1] += 1
            else:
                false_positives[int(test[0]) - 1] += 1

            hit_matrix[int(test[0]) - 1][result_k_nn[0] - 1] += 1

    for star in range(5):
        print(
            f'{star + 1}-star reviews have a precision of '
            f'{true_positives[star] / (true_positives[star] + false_positives[star])}')
    fig, ax = plt.subplots()
    im = ax.imshow(hit_matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(labels)), labels=labels)
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.xaxis.set_ticks_position('top')

    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(hit_matrix[i, j])[:5],
                    ha="center", va="center", color="w", )

    ax.set_title("Matriz de confusión")
    fig.tight_layout()

    # file_name = "asdf"
    # plt.savefig(f'ConfusionMatrix-{file_name}.png', dpi=300)
    #
    plt.show()


if __name__ == '__main__':
    df = data_clean_up()
    task1(df)
    df_training, df_test = task2(df, 0.75)
    print(df_test)
    task3(df_training, df_test)
    task4(df_training, df_test)
