import pandas as pd
from id3 import id3, predict

# Load dataset
data = pd.read_csv("dataset.csv")

# Features & Target
attributes = list(data.columns[:-1])
target = "Disease"

# Build decision tree
tree = id3(data, attributes, target)
print("Generated Decision Tree:")
print(tree)

# Test prediction
sample = {"Leaf_Color": "Yellow", "Spots": "Yes", "Wilting": "No"}
prediction = predict(tree, sample)
print(f"\nSample Input: {sample}")
print(f"Predicted Disease: {prediction}")

# User input 
user_input = {}
for attr in attributes:
    user_input[attr]= input(f"Enter value for {attr}: ")
    user_input[attr] = user_input[attr].lower().capitalize()
user_prediction = predict(tree, user_input)
print(f"\nUser Input: {user_input}")
print(f"Predicted Disease: {user_prediction}")