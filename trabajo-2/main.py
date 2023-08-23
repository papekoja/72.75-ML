import pandas as pd
import numpy as np

SEPARATORS = [',', '.', ':', '\'', '\"',
              "?", "!", "¡", "¿", "(", ")", "[", "]"]


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
        print(_features.shape)
        self.features = _features  # 2 dimensional matrix of boolean values  n x m
        self.classes = _classes  # 1 dimensional matrix of integer values  n x 1
        self.unique_values_classes, self.classes_counts = np.unique(self.classes, return_counts=True)
        self._bayes_learning_vj()

    @staticmethod
    def _relative_frequency_laplace(occurrences: int, total: int, nr_of_classes: int = 2):
        return float(occurrences + 1) / float(total + nr_of_classes)

    # the class index identifies the class in the array of unique classes
    # feature value is the value of the feature we are evaluating for, in our case its either true or false
    # feature index is the index of the column of features
    def _bayes_learning_ai_vj(self, index_of_class: int, feature_value: bool, feature_index: int) -> np.ndarray:
        _class = self.unique_values_classes[index_of_class]
        _sum_occurrences = 0
        for x in range(self.features.shape[0]):
            if self.features[x, feature_index] == int(feature_value) and self.classes[x][0] == _class:
                _sum_occurrences += 1
        return _sum_occurrences / self.classes_counts[index_of_class]

    def evaluate(self, features_in: np.ndarray):

        score = np.zeros((len(self.unique_values_classes)))

        sum_results = 0
        for class_index in range(len(score)):
            score[class_index] = 1
            for feature_index in range(self.features.shape[1]):
                score[class_index] *= self._bayes_learning_ai_vj(class_index, features_in[feature_index], feature_index)
            score[class_index] *= self.vj[class_index]
            sum_results += score[class_index]

        for x in range(len(score)):
            print(f"{self.unique_values_classes[x]} has a probability of {score[x] / sum_results}")

    # todo laplace correction for the first task
    def _bayes_learning_vj(self):
        self.vj = np.zeros((len(self.unique_values_classes)))
        print(self.vj)
        for i, _zip in enumerate(zip(self.unique_values_classes, self.classes_counts)):
            self.vj[i] = self._relative_frequency_laplace(_zip[1], len(self.classes),
                                                          len(self.unique_values_classes))
        print("vj:" + str(self.vj))


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


def dataframe_to_word_matrix(df_, word_array: TwoWayDict) -> np.ndarray:
    n_ = np.zeros((df_.shape[0], len(word_array)), np.bool)
    print(n_)
    for index, row in df_.iterrows():
        r = row["titular"]
        r = str(r)
    pass


# features:


if __name__ == '__main__':
    # df = pd.read_excel('Noticias_argentinas.xlsx')
    # word_two_way_dict = TwoWayDict(to_word_list(df))
    # print(word_two_way_dict.get_word_at(3))
    # print(word_two_way_dict.get_index_of(word_two_way_dict.get_word_at(3)))
    # dataframe_to_word_matrix(df, word_two_way_dict)

    df = pd.read_excel('PreferenciasBritanicos.xlsx')

    classifier = NaiveClassifier(df[["scones", "cerveza", "wiskey", "avena", "futbol"]].to_numpy(),
                                 df[["Nacionalidad"]].to_numpy())

    print(classifier.evaluate(np.array([1, 0, 1, 1, 0])))
