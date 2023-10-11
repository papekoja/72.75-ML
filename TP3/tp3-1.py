import numpy as np

#Pregunta 1
n_items = 100
b0 = 0.5
b1 = -1
b2 = 2

# Generate data
np.random.seed(0)  # Pour la reproductibilité
data = np.random.rand(n_items, 2) * 5

# Linearly separable labels
labels = [1 if (b0 + b1 * point[0] + b2 * point[1]) > 0 else -1 for point in data]


# Perceptrón simple
def perceptron_simple(data, labels, epochs=100, learning_rate=0.01):
    w = np.random.rand(2)
    b = 0
    
    for epoch in range(epochs):
        for point, label in zip(data, labels):
            pred = np.dot(point, w) + b # prediction
            if np.sign(pred) != label:  # bad prediction?
                w += learning_rate * label * point # update weights
                b += learning_rate * label # update bias
    return w, b




#Pregunta 3

# Generate data
data2 = np.random.rand(n_items, 2) * 5

# Linearly separable labels + noise
labels2 = []
for point in data2:
    # compute distance to line
    distance_to_line = abs(b0 + b1 * point[0] + b2 * point[1]) / np.sqrt(b1**2 + b2**2)
    
    label = 1 if (b0 + b1 * point[0] + b2 * point[1]) > 0 else -1

    # modif some label = noise
    if 0.5 < distance_to_line < 1.0:
        label *= -1
    labels2.append(label)




# Training perceptron 

w, b = perceptron_simple(data, labels)
w2, b2 = perceptron_simple(data2, labels2)




def svm_train(data, labels, learning_rate=0.01, lambda_param=0.01, epochs=1000):
    w = np.zeros(data.shape[1])
    for epoch in range(epochs):
        for i, x in enumerate(data):
            if (labels[i] * np.dot(data[i], w)) < 1: # misclassified
                w = w + learning_rate * ( (data[i] * labels[i]) + (-2 * lambda_param * w))
            else:
                w = w + learning_rate * (-2 * lambda_param * w) # update weight
    return w

# Entrainer SVM sur TP3-1
w_svm1 = svm_train(data, labels)

# Entrainer SVM sur TP3-2
w_svm2 = svm_train(data2, labels2)


# Les SVM fourniront généralement une précision meilleure ou égale par rapport au perceptron simple, surtout quand il y a des points mal classifiés ou que l'ensemble de données n'est pas complètement linéairement séparable.


import matplotlib.pyplot as plt

def plot_hyperplane(data, labels, w, b, title=""):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='jet')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1])
    yy = (-w[0] * xx - b) / w[1]
    
    plt.plot(xx, yy, 'k-')
    
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.show()



plot_hyperplane(data, labels, w, b, "Hyperplan du perceptron  TP3-1")
plot_hyperplane(data2, labels2, w2, b2, "Hyperplan du perceptron TP3-2")


plot_hyperplane(data, labels, w_svm1, 0, "Hyperplan SVM  TP3-1")


plot_hyperplane(data2, labels2, w_svm2, 0, "Hyperplan SVM  TP3-2")






