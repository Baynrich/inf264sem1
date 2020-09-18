import pandas as pd
import random
import time
import tree as my_tree
from sklearn import tree as sk_tree


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


def test_tree(impurity_measure,  nr_of_tests, X, y, pruning=False):
    avg_time = 0
    avg_score = 0
    for i in range(nr_of_tests):
        tree = my_tree.Tree()
        X, y = random_shuffle_pair(X, y)
        x_train, x_prune, x_val, x_test = split(X)
        y_train, y_prune, y_val, y_test = split(y)
        x_test = x_test + x_val
        y_test = y_test + y_val
        start = time.time()
        tree.createTree(x_train, y_train, impurity_measure, pruning, x_prune, y_prune)
        avg_score += tree.accuracy(x_test, y_test)
        avg_time += time.time() - start
    avg_time /= nr_of_tests
    avg_score /= nr_of_tests
    return avg_score, avg_time


a, b = format_data()
# random.seed(58)
X, y = random_shuffle_pair(a, b)
print("Entropy and pruning, accuracy and time: ", test_tree("entropy", 50, X, y, pruning=True))
print("Entropy, accuracy and time: ", test_tree("entropy", 50, X, y))
print("Gini and pruning, accuracy and time: ", test_tree("gini", 50, X, y, pruning=True))
print("Gini, accuracy and time: ", test_tree("gini", 50, X, y))
# x_train, x_prune, x_val, x_test = split(X)
# y_train, y_prune, y_val, y_test = split(y)
# x_train += x_prune
# y_train += x_prune
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(x_train, y_train)
# print(clf.predict(x_val))
# print(y_val)


