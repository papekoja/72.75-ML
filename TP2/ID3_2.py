import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load your dataset, let's assume you have a DataFrame called 'data'
# Replace 'target_column' with the name of your target column
data = pd.read_csv('german_credit.csv', sep=',', decimal='.')
X = data.drop('Creditability', axis=1)  # Features
y = data['Creditability']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Lists to store accuracy values
accuracy_train = []
accuracy_test = []
max_depths = range(1, 25)

# Loop through different max_depth values
for max_depth in max_depths:
    # Create a DecisionTreeClassifier with the specified max_depth
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    # Fit the classifier to the training data
    clf.fit(X_train, y_train)
    
    # Make predictions on the training data
    y_train_pred = clf.predict(X_train)
    
    # Make predictions on the test data
    y_test_pred = clf.predict(X_test)
    
    # Calculate accuracy for training and testing data
    accuracy_train.append(accuracy_score(y_train, y_train_pred) * 100)
    accuracy_test.append(accuracy_score(y_test, y_test_pred) * 100)

# Create a plot to visualize the accuracy
plt.plot(max_depths, accuracy_train, label='Train Accuracy')
plt.plot(max_depths, accuracy_test, label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs. Max Depth for Decision Tree Classifier')
plt.legend()
plt.show()
