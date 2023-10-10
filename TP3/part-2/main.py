import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import psutil
import os
import joblib
import matplotlib.pyplot as plt

p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)

def extract_patches_rgb(img, patch_size=7):
    """
    Extract local patches around each pixel in the RGB image.

    Returns:
    - np.array: Array of flattened patches.
    """
    half_size = patch_size // 2
    padded_img = cv2.copyMakeBorder(img, half_size, half_size, half_size, half_size, cv2.BORDER_REFLECT)
    patches = []

    for y in range(half_size, padded_img.shape[0] - half_size):
        for x in range(half_size, padded_img.shape[1] - half_size):
            patch = padded_img[y-half_size:y+half_size+1, x-half_size:x+half_size+1]
            patches.append(patch.flatten())

    return np.array(patches)

def classify_image(image, clf):
    # Extract features for prediction
    patches = extract_patches_rgb(image)

    # Predict categories
    predicted_categories = clf.predict(patches)

    # Map predictions to colors
    # For some reason the colors are swapped in the output image
    # But the confusion matrix is correct so it's fine (I think)
    color_map = {
        0: [255, 0, 0],  # Red = Cow
        1: [0, 255, 0],  # Green = Grass
        2: [0, 0, 255]   # Blue = Sky
    }
    colored_output = np.array([color_map[category] for category in predicted_categories])

    # Reshape the colored output to match the original image shape
    output_image = None
    if is_rgb:
        output_image = colored_output.reshape(image.shape)
    else:
        output_image = colored_output.reshape(image.shape[0], image.shape[1], 3)


    return output_image

def plot_confusion_matrix(cm, labels, Kernel, C):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap='Blues')  # Use the Blues colormap
    plt.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (C = ' + str(C) + ')')

    # Display the values in the cells of the confusion matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    #plot_confusion_matrix(confusion_mat, labels, C)
    file_path = f"TP3/part-2/results/{Kernel}/{Kernel}_C_{C}_confusion_matrix.png"
    # Save the figure to a file
    plt.savefig(file_path, dpi=300)

    # Optionally, if you don't want to display the plot after saving, you can close the figure
    plt.close()

is_rgb = True

read_mode = cv2.IMREAD_COLOR if is_rgb else cv2.IMREAD_GRAYSCALE
# Load images
vaca = cv2.imread('TP3\img/vaca.jpeg', read_mode)
pasto = cv2.imread('TP3\img\pasto.jpeg', read_mode)
cielo = cv2.imread('TP3\img\cielo.jpeg', read_mode)
cow = cv2.imread('TP3\img\cow.jpeg', read_mode)
external_cow = cv2.imread('TP3\img\external_cow.jpg', read_mode)

# Create image patches of each image
vaca_patches = extract_patches_rgb(vaca)
pasto_patches = extract_patches_rgb(pasto)
cielo_patches = extract_patches_rgb(cielo)
cow_patches = extract_patches_rgb(cow)

# Add labels to each dataset
vaca_labels = np.zeros(len(vaca_patches))
pasto_labels = np.ones(len(pasto_patches))
cielo_labels = np.ones(len(cielo_patches)) * 2

# Join dataset along with y labels
train_patches = np.concatenate((vaca_patches, pasto_patches, cielo_patches))
train_labels = np.concatenate((vaca_labels, pasto_labels, cielo_labels))

# Shuffle dataset
shuffled_indices = np.random.permutation(len(train_patches))
patches_set = train_patches[shuffled_indices]
labels_set = train_labels[shuffled_indices]

#print("Dataset size:", len(patches_set))
patches_subset = patches_set[:10000]
labels_subset = labels_set[:10000]

# Split dataset into train and test
train_quota = 0.8
train_patches = patches_subset[:int(len(patches_subset) * train_quota)]
train_labels = labels_subset[:int(len(labels_subset) * train_quota)]

test_patches = patches_subset[int(len(patches_subset) * train_quota):]
test_labels = labels_subset[int(len(labels_subset) * train_quota):]

# Store or load trained model
""" model_name = "linear_C_1"
rgb_text = "_rgb" if is_rgb else ""
filename = f"{model_name}_svm_model{rgb_text}.pkl"
joblib.dump(clf, filename)
# Load model
#clf = joblib.load(filename) """

""" # Train and test SVM for different values of C for each different kernel and store the resulting matrix
for K in ["linear", "poly", "rbf", "sigmoid"]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100, 1000]: 
        # Train the SVM
        print("Training SVM...")
        clf = svm.SVC(kernel=K, C=C) # linear, poly, rbf, sigmoid
        clf.fit(train_patches, train_labels)

        # Predict on test images and create confusion matrix
        print("Testing SVM...")
        # Make predictions on the test set
        y_pred = clf.predict(test_patches)  # Replace 'clf' with the name of your trained SVM classifier

        # Create a confusion matrix
        confusion_mat = confusion_matrix(test_labels, y_pred)

        # Plot the confusion matrix
        labels = ['Vaca', 'Pasto', 'Cielo']
        
        plot_confusion_matrix(confusion_mat, labels, K, C) """

# Train the SVM
print("Training SVM...")
clf = svm.SVC(kernel="poly") # linear, poly, rbf, sigmoid
clf.fit(train_patches, train_labels)

# Predict on a new image
print("Predicting on a new image...")

output_image = classify_image(external_cow, clf)

# Show the output image
output_image = np.clip(output_image, 0, 255).astype(np.uint8)
cv2.imshow('Output', output_image)
cv2.waitKey(0)