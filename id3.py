import numpy as np
import pandas as pd
from collections import Counter

# Function to calculate entropy
def entropy(data, target_attribute):
    values = data[target_attribute].unique()
    entropy_val = 0
    for v in values:
        p = len(data[data[target_attribute] == v]) / len(data)
        entropy_val -= p * np.log2(p)
    return entropy_val

# Function to calculate information gain
def information_gain(data, attribute, target_attribute):
    total_entropy = entropy(data, target_attribute)
    values = data[attribute].unique()
    weighted_entropy = 0
    for v in values:
        subset = data[data[attribute] == v]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset, target_attribute)
    return total_entropy - weighted_entropy

# Recursive function to build decision tree
def id3(data, attributes, target_attribute):
    # If all target values are same → return that class
    if len(data[target_attribute].unique()) == 1:
        return data[target_attribute].iloc[0]
    
    # If no attributes left → return most common target
    if len(attributes) == 0:
        return Counter(data[target_attribute]).most_common(1)[0][0]
    
    # Find best attribute
    gains = [information_gain(data, attr, target_attribute) for attr in attributes]
    best_attr = attributes[np.argmax(gains)]
    
    # Build tree
    tree = {best_attr: {}}
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        if subset.empty:
            tree[best_attr][value] = Counter(data[target_attribute]).most_common(1)[0][0]
        else:
            new_attrs = [a for a in attributes if a != best_attr]
            subtree = id3(subset, new_attrs, target_attribute)
            tree[best_attr][value] = subtree
    return tree

# Prediction function
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    value = sample.get(attribute)
    if value in tree[attribute]:
        return predict(tree[attribute][value], sample)
    else:
        return None
