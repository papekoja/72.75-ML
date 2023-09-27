import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# import the data and do a train test split
data = pd.read_csv('german_credit.csv', sep=',', decimal='.')
train, test = train_test_split(data, test_size=0.1, random_state=38)

# 1. entropy formula
def entropy(data):
    value_counts = data.value_counts() / len(data)
    entropy = -sum(p * math.log2(p) for p in value_counts)
    return entropy

# 2. information gain
def information_gain(data, attribute_name, target_name):
    # Calculate the entropy of the entire dataset
    total_entropy = entropy(data[target_name])
    
    # Group the data by the unique values of the attribute
    grouped_data = data.groupby(attribute_name)
    
    # Calculate the weighted average of entropies for each group
    weighted_entropy = sum((len(group) / len(data)) * entropy(group[target_name]) for _, group in grouped_data)
    
    # Calculate information gain
    info_gain = total_entropy - weighted_entropy
    
    return info_gain

# 3. Find the best information gain to split on
def best_information_gain(data, target):
    # Calculate the information gain for each attribute
    info_gains = {col: information_gain(data, col, target) for col in data.columns if col != target}

    # Return the attribute with the highest information gain
    return max(info_gains, key=info_gains.get)






# 4. make the treeeeee

class TreeNode:
    def __init__(self, data):
        self.data = data  # Data associated with the node (attribute or class label)
        self.children = {}  # Dictionary to store child nodes

# Define a function to build a decision tree recursively
def build_tree(data, target_attribute):
    attributes = list(data.columns)
    
    # If all examples belong to the same class, create a leaf node with that class
    if len(data[target_attribute].unique()) == 1:
        return TreeNode(data[target_attribute].iloc[0])
    
    # If there are no more attributes to split on, create a leaf node with the majority class
    if len(attributes) == 0:
        majority_class = data[target_attribute].mode()[0]
        return TreeNode(majority_class)
    
    # Calculate the Information Gain for each attribute and choose the one with the highest IG
    max_info_gain = -1
    best_attribute = None
    for attribute in attributes:
        ig = information_gain(data, attribute, target_attribute)  # Use the information_gain function from the previous response
        if ig > max_info_gain:
            max_info_gain = ig
            best_attribute = attribute
    
    # Create a new tree node with the best attribute
    tree = TreeNode(best_attribute)
    
    # Remove the best attribute from the list of attributes
    attributes.remove(best_attribute)
    
    # Recursively build the tree for each value of the best attribute
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        if subset.empty:
            # If a subset is empty, create a leaf node with the majority class of the parent node
            majority_class = data[target_attribute].mode()[0]
            tree.children[value] = TreeNode(majority_class)
        else:
            # Recursively build the subtree
            tree.children[value] = build_tree(subset, target_attribute)
    
    return tree


def predict(tree, instance):
    # Start at the root node
    current_node = tree
    
    # Traverse the tree until a leaf node is reached
    while isinstance(current_node.data, str):  # Check if the current node is an attribute (non-leaf)
        # Get the attribute value of the current instance
        attribute_value = instance[current_node.data]
        
        # If the attribute value is in the tree's children, move to the corresponding child node
        if attribute_value in current_node.children:
            current_node = current_node.children[attribute_value]
        else:
            # If the attribute value is not in the tree's children, return a default value or handle the case as needed
            return None  # You can customize this part based on your application
    
    # The current node is a leaf node, so return its class label
    return current_node.data







decision_tree = build_tree(train, 'Creditability')

right, wrong = 0, 0

for i in range(100):
    test_instance = test.iloc[i]
    instance_to_predict = {test_instance.index[i]: test_instance[i] for i in range(len(test_instance))}
    predicted_class = predict(decision_tree, instance_to_predict)

    if test_instance == predicted_class: 
        right += 1
    else: 
        wrong +=1
    
  

    # print(f'Predicted Class: {predicted_class}\nActual Class:    {test_instance["Creditability"]}\n')

    print(f"right: {right}\wrong: {wrong}")


