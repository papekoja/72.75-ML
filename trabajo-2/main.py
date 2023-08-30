import pandas as pd
import numpy as np
import threading

SEPARATORS = [',', '.', ':', '\'', '\"',
              "?", "!", "¡", "¿", "(", ")", "[", "]"]


class MostlyEmptyBinaryMatrix():
    def __init__(self, rows_: int):
        self.rows_amount = rows_
        self.rows = np.array()


class TwoWayDict:
    def __init__(self, word_list: []):
        self.words = word_list
        self.reverse = dict()
        for count, value in enumerate(word_list):
            self.reverse[value] = count

        if len(self.words) != len(self.reverse):
            print("nonononono")

    def get_word_at(self, index: int):
        return self.words[index]

    def get_index_of(self, word: str):
        return self.reverse[word]

    def __len__(self):
        return len(self.words)


class NaiveClassifier:
    def __init__(self, _features: np.ndarray, _classes: np.ndarray):

        self.features = _features  # 2 dimensional matrix of boolean values  n x m
        self.classes = np.zeros(_classes.shape, np.uint8)  # 1 dimensional matrix of integer values  n x 1

        unique_values_classes_names, self.classes_counts = np.unique(_classes, return_counts=True)
        self.classes_int_dict = TwoWayDict(unique_values_classes_names.tolist())
        for x in range(_classes.shape[0]):
            self.classes[x][0] = self.classes_int_dict.get_index_of(_classes[x][0])

        self.unique_values_classes = np.zeros(unique_values_classes_names.shape)
        for x in range(self.unique_values_classes.shape[0]):
            self.unique_values_classes[x] = self.classes_int_dict.get_index_of(unique_values_classes_names[x])

        self.features_classes = np.concatenate((self.features, self.classes), axis=1)
        print(f"features shape: {self.features.shape}")
        print(f"classes shape: {self.classes.shape}")
        print(f"features + classes shape: {self.features_classes.shape}")

        self.ai_vj = np.zeros((len(unique_values_classes_names), self.features.shape[1], 2), float)

        self._bayes_learning_vj()
        self._precalculate()

    @staticmethod
    def _relative_frequency_laplace(occurrences: int, total: int, nr_of_classes: int = 2):
        return float(occurrences + 1) / float(total + nr_of_classes)

    # the class index identifies the class in the array of unique classes
    # feature value is the value of the feature we are evaluating for, in our case its either true or false
    # feature index is the index of the column of features
    def _bayes_learning_ai_vj(self, index_of_class: int, feature_index: int, feature_value: bool) -> float:
        _class = self.unique_values_classes[index_of_class]
        _sum_occurrences = 0
        for x in range(self.features.shape[0]):
            if self.features[x, feature_index] == int(feature_value) and self.classes[x][0] == _class:
                _sum_occurrences += 1
        return self._relative_frequency_laplace(_sum_occurrences, self.classes_counts[index_of_class],
                                                len(self.classes_counts))

    def _bayes_learning_ai_vj_mask(self, index_of_class: int, feature_index: int, feature_value: bool) -> float:
        _class = self.unique_values_classes[index_of_class]
        mask = (self.features_classes[:, feature_index] == int(feature_value))
        mask_2 = (self.features_classes[:, self.features.shape[1]] == int(_class))

        _sum_occurrences = len(self.features_classes[mask & mask_2, :])

        return self._relative_frequency_laplace(_sum_occurrences, self.classes_counts[index_of_class],
                                                len(self.classes_counts))

    def _calculate_features_for_class(self, c: int):
        last_update = 0
        for f in range(self.features.shape[1]):
            temp = f / self.features.shape[1]
            if (temp - last_update) > 0.05:
                last_update = temp
                print(f"|{c}|", end="")
            self.ai_vj[c, f, 0] = self._bayes_learning_ai_vj_mask(c, f, False)
            self.ai_vj[c, f, 1] = self._bayes_learning_ai_vj_mask(c, f, True)

    def _precalculate(self):
        for c in range(len(self.unique_values_classes)):
            print("calculating class " + str(c))
            self._calculate_features_for_class(c)
        print("completed precalculation")

    def _precalculate_multi_thread(self):
        thread_pool = []
        for c in range(len(self.unique_values_classes)):
            print("calculating class " + str(c))
            t = threading.Thread(target=self._calculate_features_for_class, args=(c,))
            t.start()
            thread_pool.append(t)

        for t in thread_pool:
            t.join()

        print("completed precalculation")

    def evaluate(self, features_in: np.ndarray) -> dict:

        score = np.zeros((len(self.unique_values_classes)))

        sum_results = 0
        for class_index in range(len(score)):
            # print("analyzing class: " + str(class_index))
            score[class_index] = 1
            for feature_index in range(self.features.shape[1]):
                score[class_index] *= self.ai_vj[class_index, feature_index, features_in[feature_index]]
            score[class_index] *= self.vj[class_index]
            sum_results += score[class_index]

        results_ = dict()
        for x in range(len(score)):
            results_[self.classes_int_dict.get_word_at(int(self.unique_values_classes[x]))] = score[x] / sum_results
            # print(
            #     f"'{self.classes_int_dict.get_word_at(int(self.unique_values_classes[x]))}' has a probability of {score[x]} and a normalized "
            #     f"portability of:  {score[x] / sum_results}")

        return results_

    def _bayes_learning_vj(self):
        self.vj = np.zeros((len(self.unique_values_classes)))
        for i, _zip in enumerate(zip(self.unique_values_classes, self.classes_counts)):
            self.vj[i] = self._relative_frequency_laplace(_zip[1], len(self.classes),
                                                          len(self.unique_values_classes))


def row_to_word_arr(r: str) -> []:
    # replace seperators with " "
    for s in SEPARATORS:
        r = r.replace(s, " ")

    # replace "  " with " " unitl only single spaces are left in between the words
    while "  " in r:
        r = r.replace("  ", " ")

    # words to lowercase
    r = r.lower()
    # split words and add them to the word set
    return r.split(" ")


# returns a list of words in the dataframe
def to_word_list(df_) -> []:
    word_set = set()
    for index, row in df_.iterrows():
        r = row["titular"]
        r = str(r)
        word_set.update(row_to_word_arr(r))
    return list(word_set)


# !! needs some memory!!
def dataframe_to_word_matrix(df_, word_array: TwoWayDict) -> np.ndarray:
    n_ = np.zeros((df_.shape[0], len(word_array)), np.uint8)
    print(n_)
    for index, row in enumerate(df_.iterrows()):
        r = row[1]["titular"]
        words_of_row = row_to_word_arr(str(r))
        for w in words_of_row:
            word_index = word_array.get_index_of(w)
            n_[index][word_index] = 1
    print(n_)
    return n_


# features:


def get_highest_score(scores: dict):
    res_highest = ""
    score_highest = 0
    for k in scores.keys():
        if scores[k] > score_highest:
            res_highest = k
            score_highest = scores[k]
    return res_highest


def task_1():
    df = pd.read_excel('PreferenciasBritanicos.xlsx')

    classifier = NaiveClassifier(df[["scones", "cerveza", "wiskey", "avena", "futbol"]].to_numpy(),
                                 df[["Nacionalidad"]].to_numpy())

    classifier.evaluate(np.array([1, 0, 1, 1, 0]))
    classifier.evaluate(np.array([0, 1, 1, 0, 1]))


def task_2(split_training: float, eliminate_perc=0.0):
    selected_categories = ['Internacional',
                           'Nacional',
                           # 'Destacadas',
                           'Deportes',
                           'Salud',
                           # 'Ciencia y Tecnologia',
                           # 'Entretenimiento',
                           # 'Economia',
                           ]

    df = pd.read_excel('Noticias_argentinas.xlsx')[["titular", "fuente", "categoria"]]
    df = df.loc[df['categoria'].isin(selected_categories)]
    print(df)
    categories_unique = df['categoria'].unique()
    df_training = pd.DataFrame(columns=df.columns)
    df_test = pd.DataFrame(columns=df.columns)
    for c in categories_unique:
        df_temp = df.query(f'categoria == "{c}"')
        rows_temp = df_temp.shape[0]

        # remove data for faster testing
        df_temp = df_temp.iloc[:int(rows_temp * (1 - eliminate_perc)), :]
        rows_temp = df_temp.shape[0]

        split = int(rows_temp * split_training)
        df_training_temp = df_temp.iloc[:split, :]
        df_test_temp = df_temp.iloc[split:, :]
        print(f"dataset length for category: {c}: training: {df_training_temp.shape[0]}, test: {df_test_temp.shape[0]}")
        df_training = pd.concat(
            [df_training, df_training_temp],
            axis=0)

        df_test = pd.concat(
            [df_test, df_test_temp],
            axis=0
        )

    print("test:")
    print(df_test)

    print("training:")
    print(df_training)

    word_two_way_dict = TwoWayDict(to_word_list(df))
    word_matrix = dataframe_to_word_matrix(df_training, word_two_way_dict)

    classifier = NaiveClassifier(word_matrix, df_training[["categoria"]].to_numpy())

    test_data_word_matrix = dataframe_to_word_matrix(df_test, word_two_way_dict)

    hits = 0
    for index, row in enumerate(df_test.iterrows()):
        result = classifier.evaluate(test_data_word_matrix[index].ravel())
        category = row[1].at['categoria']
        highest_cat = get_highest_score(result)
        if category == highest_cat:
            hits += 1

    print(f"correctly classified: {100*(hits / test_data_word_matrix.shape[0] )}")

if __name__ == '__main__':
    # task_1()
    task_2(0.9)
