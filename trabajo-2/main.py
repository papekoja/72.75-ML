import pandas as pd
import numpy as np

SEPARATORS = [',', '.', ':', '\'', '\"',
              "?", "!", "¡", "¿", "(", ")", "[", "]"]


class NaiveClassifier:
    def __init__(self, _features: np.ndarray, _classes: np.ndarray):
        self.features = _features  # 2 dimensional matrix of boolean values  n x m
        self.classes = _classes  # 1 dimensional matrix of integer values  n x 1
        self.unique_values_classes, self.classes_counts = np.unique(self.classes, return_counts=True)
        print("classes:")
        print(self.classes)
        print("\nfeatures")
        print(self.features)

        self._bayes_learning_vj()

    @staticmethod
    def _relative_frequency_laplace(occurrences: int, total: int, nr_of_classes: int = 2):
        return float(occurrences + 1) / float(total + nr_of_classes)

    def _bayes_learning_ai_vj(self) -> np.ndarray:
        pass

    def _bayes_learning_vj(self):
        self.vj = np.zeros((len(self.unique_values_classes), 1))
        print(self.vj)
        for i, _zip in enumerate(zip(self.unique_values_classes, self.classes_counts)):
            self.vj[i] = self._relative_frequency_laplace(_zip[1], len(self.classes),
                                                          len(self.unique_values_classes))



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
