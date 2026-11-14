# Switzerland (Sigmoid LMML 2026, Easy)

Fits a `DecisionTreeClassifier(random_state=0)` that predicts `slab_released`.

The word is read from the order of the tree's split questions: a pre-order traversal from the root, taking the first letter of each split feature. `main.py` reads `test-input.csv` and writes `test-output.txt`.
