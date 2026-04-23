# 🌱 CropDoc

**CropDoc** is a lightweight plant disease detection system powered by the **ID3 decision tree algorithm**.  
It helps in identifying common plant diseases based on leaf symptoms like color, spots, and wilting.  
Designed for learning and scalable to real-world agricultural datasets.

---

## 🚀 Features
- 🌿 Built with **ID3** (Entropy & Information Gain)  
- 📊 Small sample dataset included (`dataset.csv`)  
- ⚡ Scalable to **large datasets** using `scikit-learn`  
- 🔎 Human-readable decision tree rules  

---

## 📂 Project Structure
CropDoc/
│── dataset.csv # sample dataset
│── id3.py # ID3 algorithm implementation
│── main.py # training & testing entry point
│── README.md # documentation

---

## 🛠️ Tech Stack

  - Python 3
  - pandas → data handling
  - numpy → entropy & math operations
  - scikit-learn → large dataset scaling (optional)

---

## ⚙️ Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/CropDoc.git
cd CropDoc
pip install pandas numpy scikit-learn
```

## ▶️ Usage :

Run with the sample dataset:
```bash
python main.py
```

## ✅ Example Output
```bash
    Generated Decision Tree:
    {'Spots': {'Yes': 'Leaf_Spot', 'No': 'Wilt'}}
    
    Sample Input: {'Leaf_Color': 'Yellow', 'Spots': 'Yes', 'Wilting': 'No'}
    Predicted Disease: Leaf_Spot
```
---

## 🌍Future Goals

Replace the small dataset with a larger one (e.g., PlantVillage dataset).
and use scikit-learn for efficient training:
        from sklearn.tree import DecisionTreeClassifier.        
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=10).
        clf.fit(X_train, y_train).
        print("Accuracy:", clf.score(X_test, y_test)).

---

## 📖 Learning Outcomes
 🌟 Understand Entropy & Information Gain.
 🌟 Implement ID3 algorithm from scratch.
 🌟 Learn how to handle unseen data with defaults.
 🌟 Scale to real agricultural datasets.

---

## 📜 License

This project is licensed under the MIT License © 2025.
You are free to use, modify, and distribute with attribution.
---
