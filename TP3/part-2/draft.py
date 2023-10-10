import random
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def add_gaussian_noise_to_grayscale(image, mean=0, sigma=25):
    """Add Gaussian noise to a grayscale image.
    
    Args:
        image (np.array): Original grayscale image.
        mean (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        
    Returns:
        np.array: Grayscale image with added Gaussian noise.
    """
    row, col = image.shape
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy_image = image + gauss
    
    # Ensure the pixel values are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def create_gaussian_noise_dataset(image, n_samples=100, mean=0, sigma=25):
    """Create a dataset of noisy images.
    
    Args:
        image (np.array): Original grayscale image.
        n_samples (int): Number of noisy images to generate.
        mean (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
        
    Returns:
        np.array: Dataset of noisy images.
    """
    dataset = []
    for _ in range(n_samples):
        noisy_image = add_gaussian_noise_to_grayscale(image, mean, sigma)
        dataset.append(noisy_image)
    return np.array(dataset)

def random_crop(image, crop_height, crop_width):
    """
    Randomly crop an image to the specified height and width.
    
    Args:
    - image (np.array): Original image.
    - crop_height (int): Height of the cropped region.
    - crop_width (int): Width of the cropped region.
    
    Returns:
    - np.array: Cropped image.
    """
    if crop_height > image.shape[0] or crop_width > image.shape[1]:
        raise ValueError("Cropped dimensions are larger than image dimensions!")
    
    y = np.random.randint(0, image.shape[0] - crop_height)
    x = np.random.randint(0, image.shape[1] - crop_width)

    cropped_image = image[y:y+crop_height, x:x+crop_width]

    return cropped_image

# Load images
vaca = cv2.imread('TP3\img/vaca.jpeg', cv2.IMREAD_GRAYSCALE)
pasto = cv2.imread('TP3\img\pasto.jpeg', cv2.IMREAD_GRAYSCALE)
cielo = cv2.imread('TP3\img\cielo.jpeg', cv2.IMREAD_GRAYSCALE)
cow = cv2.imread('TP3\img\cow.jpeg', cv2.IMREAD_GRAYSCALE)

# Create a dataset of noisy images
print("Creating dataset of noisy images..."	)
vaca_dataset = create_gaussian_noise_dataset(vaca)
pasto_dataset = create_gaussian_noise_dataset(pasto)
cielo_dataset = create_gaussian_noise_dataset(cielo)

# Random crop all images in the datasets
print("Randomly cropping images...")
vaca_cropped_dataset = []
for i in range(len(vaca_dataset)):
    vaca_cropped_dataset.append(random_crop(vaca_dataset[i], 140, 140))

pasto_cropped_dataset = []
for i in range(len(pasto_dataset)):
    pasto_cropped_dataset.append(random_crop(pasto_dataset[i], 140, 140))

cielo_cropped_dataset = []
for i in range(len(cielo_dataset)):
    cielo_cropped_dataset.append(random_crop(cielo_dataset[i], 140, 140))

# Flatten all the images in the datasets
vaca_cropped_dataset = [image_vector.flatten() for image_vector in vaca_cropped_dataset]
pasto_cropped_dataset = [image_vector.flatten() for image_vector in pasto_cropped_dataset]
cielo_cropped_dataset = [image_vector.flatten() for image_vector in cielo_cropped_dataset]

# Join dataset along with y labels
x = np.concatenate((vaca_cropped_dataset, pasto_cropped_dataset, cielo_cropped_dataset))
y = np.concatenate((np.zeros(len(vaca_cropped_dataset)), np.ones(len(pasto_cropped_dataset)), np.ones(len(cielo_cropped_dataset)) * 2))

X, Y = shuffle(x, y, random_state=42)


# Split dataset into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

