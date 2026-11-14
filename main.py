import pandas as pd
from sklearn.tree import DecisionTreeClassifier

TARGET = "slab_released"

df = pd.read_csv("test-input.csv")
y = df[TARGET] if TARGET in df.columns else df.iloc[:, -1]
X = df.drop(columns=[y.name])

tree = DecisionTreeClassifier(random_state=0).fit(X, y).tree_
names = list(X.columns)

# Read the split questions in the order the tree asks them (root first,
# pre-order traversal); the first letter of each split feature is one letter.
letters = []
stack = [0]
while stack:
    n = stack.pop()
    left, right = tree.children_left[n], tree.children_right[n]
    if left == -1:
        continue
    letters.append(names[tree.feature[n]][0].upper())
    stack.append(right)
    stack.append(left)

with open("test-output.txt", "w") as f:
    f.write("SIGMOID_" + "".join(letters) + "\n")
