import json
import numpy as np
from main import TwoWayDict
import matplotlib as mpl
import matplotlib.pyplot as plt


def loadFile(filename: str):
    with open(filename, 'r') as file_object:
        jsonStr = file_object.read()
    aList = json.loads(jsonStr)

    return aList

def subtask_1_2_3():
  # confusion matrix
    file_name = "90-Deportes-Salud-Ciencia y Tecnologia-Entretenimiento.json"
    data = loadFile(file_name)
    classes_amount = len(data[0][0])
    print(classes_amount)
    hit_matrix = np.zeros((classes_amount, classes_amount))

    t = TwoWayDict(list(data[0][0].keys()))

    # sum entries
    for d in data:
        result, highest_cat, category = d
        hit_matrix[int((t.get_index_of(category)))][int(t.get_index_of(highest_cat))] += 1

    true_pos = []
    true_neg = []
    false_pos = []
    false_neg = []

    sum_total = 0
    for x in range(classes_amount):
        sum_total += hit_matrix[x].sum()

    for x in range(classes_amount):
        true_pos.append(hit_matrix[x][x])

        colum_sum = 0
        for y in range(classes_amount):
            colum_sum += hit_matrix[y][x]

        false_pos.append(colum_sum - hit_matrix[x][x])
        false_neg.append(hit_matrix[x].sum() - hit_matrix[x][x])

        true_neg = sum_total - false_pos - false_neg - true_pos

    accuracy = []
    precision = []
    tasa_verd_pos = []
    tasa_fals_pos = []
    f1_score = []

    print(f"accuracy: ")
    for x in range(classes_amount):
        accuracy.append((true_pos[x] + true_neg[x]) / (true_pos[x] + true_neg[x] + false_neg[x] + false_pos[x]))
        print(f"{t.get_word_at(x)} = {str(accuracy[x])[:6]}")

    print(f"precision: ")
    for x in range(classes_amount):
        precision.append(true_pos[x] / (true_pos[x] + false_pos[x]))
        print(f"{t.get_word_at(x)} = {str(precision[x])[:6]}")

    print(f"tasa de verdaderos prositivos: ")
    for x in range(classes_amount):
        tasa_verd_pos.append(true_pos[x] / (true_pos[x] + false_neg[x]))
        print(f"{t.get_word_at(x)} = {str(tasa_verd_pos[x])[:6]}")

    print(f"tasa de falsos prositivos: ")
    for x in range(classes_amount):
        tasa_fals_pos.append(false_pos[x] / (false_pos[x] + true_neg[x]))
        print(f"{t.get_word_at(x)} = {str(tasa_fals_pos[x])[:6]}")

    print("f1_score")
    for x in range(classes_amount):
        f1_score.append((2 * precision[x] * tasa_verd_pos[x]) /  (precision[x] + tasa_verd_pos[x]))
        print(f"{t.get_word_at(x)} = {str(f1_score[x])[:6]}")

    # normalize results
    for x in range(classes_amount):
        _sum = hit_matrix[x].sum()
        hit_matrix[x] /= _sum

    fig, ax = plt.subplots()
    im = ax.imshow(hit_matrix)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(t.words)), labels=t.words)
    ax.set_yticks(np.arange(len(t.words)), labels=t.words)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.xaxis.set_ticks_position('top')

    # Loop over data dimensions and create text annotations.
    for i in range(len(t.words)):
        for j in range(len(t.words)):
            text = ax.text(j, i, str(hit_matrix[i, j])[:5],
                           ha="center", va="center", color="w", )

    ax.set_title("Matriz de confusión")
    fig.tight_layout()
    plt.savefig(f'ConfusionMatrix-{file_name}.png', dpi=300)
    plt.show()


def _roc(acceptance_threshold: float, data, category_to_analyse: str) -> tuple:
    true_positives = 0 # verdadero postive vp
    true_negative = 0 # verdadero negative vn

    false_positive = 0 # falso positive fp
    false_negative = 0 # falso negativo fn

    for d in data:
        result, highest_cat, category = d
        predicted_percentage = result[category_to_analyse]
        if (predicted_percentage > acceptance_threshold): #positive prediccion
            if (category == category_to_analyse):# real positive
                true_positives += 1
            else: # false positive
                false_positive += 1
        else: #negative prediccion
            if (category == category_to_analyse):# real positive
                false_negative += 1
            else: # false positive
                true_negative += 1
                
    TVP = true_positives / (true_positives + false_negative)
    TFP = false_positive / (false_positive + true_negative)
    return TVP,TFP



def subtask_roc(file_name : str, samples: int):
    data = loadFile(file_name)


    categories = list(data[0][0].keys())
    for category in categories:
        results_TVP = []
        results_TFP = []
        for i in range(samples + 1):
            p = i / samples
            TVP,TFP = _roc(p,data,category)
            results_TVP.append(TVP)
            results_TFP.append(TFP)    

        print(category)
        plt.plot(results_TFP,results_TVP, label = category)
    plt.legend(loc="lower right")
    plt.show()

if __name__ == '__main__':
    # subtask_1_2_3()
    subtask_roc("80-Deportes-Salud-Ciencia y Tecnologia-Entretenimiento.json", 100)