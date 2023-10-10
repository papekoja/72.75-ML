import numpy as np
import cv2
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
import psutil
import os
from sklearn.externals import joblib

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
    # For this example, let's use red, green, and blue for the three categories
    color_map = {
        0: [255, 0, 0],  # Red
        1: [0, 255, 0],  # Green
        2: [0, 0, 255]   # Blue
    }
    colored_output = np.array([color_map[category] for category in predicted_categories])

    # Reshape the colored output to match the original image shape
    output_image = colored_output.reshape(image.shape)

    return output_image

# Load images
vaca = cv2.imread('TP3\img/vaca.jpeg', cv2.IMREAD_GRAYSCALE)
pasto = cv2.imread('TP3\img\pasto.jpeg', cv2.IMREAD_GRAYSCALE)
cielo = cv2.imread('TP3\img\cielo.jpeg', cv2.IMREAD_GRAYSCALE)
cow = cv2.imread('TP3\img\cow.jpeg', cv2.IMREAD_GRAYSCALE)

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

# Split dataset into train and test
train_quota = 0.8
train_patches = patches_set[:int(len(patches_set) * train_quota)]
train_labels = labels_set[:int(len(labels_set) * train_quota)]

test_patches = patches_set[int(len(patches_set) * train_quota):]
test_labels = labels_set[int(len(labels_set) * train_quota):]

# Train the SVM
print("Training SVM...")
clf = svm.SVC(kernel='linear') # linear, poly, rbf, sigmoid
clf.fit(train_patches, train_labels)

# Store trained model
model_name = "linear_C_1"
filename = f"{model_name}_svm_model.pkl"
joblib.dump(clf, filename)

# Predict on test images and create confusion matrix
print("Testing SVM...")
# Make predictions on the test set
y_pred = clf.predict(test_patches)  # Replace 'clf' with the name of your trained SVM classifier

# Create a confusion matrix
confusion_mat = confusion_matrix(test_labels, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_mat)

""" # Predict on a new image
print("Predicting on a new image...")

output_image = classify_image(cow, clf)

# Show the output image
cv2.imshow('Output', output_image)
cv2.waitKey(0) """