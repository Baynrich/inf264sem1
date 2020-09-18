import pandas as pd
import random
import tree as my_tree


def format_data(filename='data_banknote_authentication.txt'):
    df = pd.read_csv('data_banknote_authentication.txt')
    y_df = df['Label']
    X_df = df[['A', 'B', 'C', 'D']]
    X = []
    y = []
    for i in range(len(X_df)):
        X.append(X_df.iloc[i].to_numpy().tolist())
        y.append(y_df.iloc[i])
    return X, y


def random_shuffle_pair(a, b):
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def split(X):
    train, prune, val,  test = X[:int(len(X) * 0.6)], X[int(len(X) * 0.6): int(len(X) * 0.7)],\
                               X[int(len(X) * 0.7): int(len(X) * 0.85)], X[int(len(X) * 0.85):]
    return train, prune, val, test


def train_and_test_get_best_tree(X,y):
    x_train, x_prune, x_val, x_test = split(X)
    y_train, y_prune, y_val, y_test = split(y)
    tree1, tree2, tree3, tree4 = my_tree.Tree(), my_tree.Tree(), my_tree.Tree(), my_tree.Tree()
    tree1.createTree(x_train, y_train)
    tree2.createTree(x_train, y_train, impurity_measure="gini")
    tree3.createTree(x_train, y_train, prune=True, x_prune=x_prune, y_prune=y_prune)
    tree4.createTree(x_train, y_train, impurity_measure="gini", prune=True, x_prune=x_prune, y_prune=y_prune)
    entropy = tree1.accuracy(x_val, y_val)
    gini = tree2.accuracy(x_val, y_val)
    entropy_pruned = tree3.accuracy(x_val, y_val)
    gini_pruned = tree4.accuracy(x_val, y_val)
    trees = [tree1, tree2, tree3, tree4]
    tree_accuracies = [entropy, gini, entropy_pruned, gini_pruned]
    tree_names = ["Entropy", "Gini", "Entropy with pruning", "Gini with pruning"]
    for i in range(4):
        if tree_accuracies[i] == max(tree_accuracies):
            print(tree_names[i], " has the highest accuracy with : " , tree_accuracies[i])
            print(trees[i].accuracy(x_test, y_test), " is the accuracy of this model with the testing set.")
            return trees[i]


a, b = format_data()
# random.seed(46)           ##If you want to use a fixed seed
X, y = random_shuffle_pair(a, b)
best_tree = train_and_test_get_best_tree(X, y)





