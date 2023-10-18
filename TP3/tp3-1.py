import numpy as np
import matplotlib.pyplot as plt



def calcular_error(X, y, w):
    pred = np.sign(np.dot(X, w))
    error = np.sum(pred != y)
    return error

def perceptron_simple(X, y, COTA=1000):
    p, n = X.shape
    w = np.zeros(n)
    b = 0  
    w_min = np.copy(w)
    b_min = b
    error_min = float('inf')
    
    for i in range(COTA):
        idx = np.random.randint(0, p)  # random sample
        xi, yi = X[idx], y[idx]

        h = np.dot(xi, w) + b
        if np.sign(h) != yi:
            delta_w = yi * xi
            w += delta_w
            b += yi  # update bias

            error = calcular_error(np.hstack((np.ones((p, 1)), X)), y, np.hstack(([b], w)))
            if error < error_min:
                error_min = error
                w_min = np.copy(w)
                b_min = b  # update minimum biais

    return w_min, b_min  



def perceptron_optimal(data, labels, epochs=100, learning_rate=0.01):
    w = np.random.rand(2)
    b = 0
    
    for epoch in range(epochs):
        total_error = 0
        for point, label in zip(data, labels):
            pred = np.dot(point, w) + b
            if np.sign(pred) != label: # misclassified point
                total_error += abs(label - pred) # error is the distance from the point to the line
                w += learning_rate * label * point # update weights
                b += learning_rate * label  # update bias
        if total_error == 0:  # if no error, stop
            break
    return w, b







def plot_comparison(data, labels, w_normal, b_normal, w_optimized, b_optimized):
    plt.figure(figsize=(10, 7))

    # Plotting the data points
    plt.scatter(data[labels == 1][:, 0], data[labels == 1][:, 1], c='blue', label='Class 1')
    plt.scatter(data[labels == -1][:, 0], data[labels == -1][:, 1], c='red', label='Class -1')

    # Perceptron normal line
    x_values = np.linspace(min(data[:,0]), max(data[:,0]), 100)
    y_values_normal = (-w_normal[0] * x_values - b_normal) / w_normal[1]
    plt.plot(x_values, y_values_normal, 'g--', label='Perceptron normal')

    # Perceptron optimized line
    y_values_optimized = (-w_optimized[0] * x_values - b_optimized) / w_optimized[1]
    plt.plot(x_values, y_values_optimized, 'c-', label='Perceptron optimized')

    plt.title('Comparison between Normal and Optimized Perceptron')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()


n_items = 500
b0 = 0.5
b1 = -1
b2 = 2

# Generate data
np.random.seed(2300)  
data = np.random.rand(n_items, 2) * 5

# Linearly separable labels
labels = np.array([1 if (b0 + b1 * point[0] + b2 * point[1]) > 0 else -1 for point in data])


w_1,b_1 = perceptron_simple(data, labels)
w_optimal, b_optimal = perceptron_optimal(data, labels)

#Results
plot_comparison(data, labels, w_1,b_1, w_optimal, b_optimal)




