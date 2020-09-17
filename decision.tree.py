import numpy as np
import functools as ft
import pandas as pd
import random


class Tree:

    def splitMatrixByAverage(self, X, y, xCol, splitBy):
        curXCol = [xRow[xCol] for xRow in X]
        yAboveOrEqual = []
        yBelow = []
        for i in range(len(y)):
            if (curXCol[i] >= splitBy):
                yAboveOrEqual.append(y[i])
            else:
                yBelow.append(y[i])
        return [yAboveOrEqual, yBelow]

    def getEnt(self, list):
        probZero = len([val for val in list if val == 0]) / len(list)
        if probZero == 1 or probZero == 0:
            return 0
        return (-1) * ((probZero * np.log2(probZero)) + ((1 - probZero) * np.log2(1 - probZero)))

    def splitByEntropy(self, X, y):
        # Full group entropy
        curEnt = self.getEnt(y)
        splitBys = []
        IGs = []
        # Calculate all information gain values
        for xCol in range(len(X[0])):
            splitBy = np.average([xRow[xCol] for xRow in X])
            splitBys.append(splitBy)
            ySplit = self.splitMatrixByAverage(X, y, xCol, splitBy)
            entAboveOrEqual = self.getEnt(ySplit[0])
            entBelow = self.getEnt(ySplit[1])
            conditionalEnt = (len(ySplit[0]) / len(y)) * entAboveOrEqual + (len(ySplit[1]) / len(y)) * entBelow;
            IGs.append(curEnt - conditionalEnt)
        # Save index of column used to split matrix, EG. column with highest information gain
        self.splitter = IGs.index(max(IGs))
        self.splitBy = splitBys[IGs.index(max(IGs))]

    def getGINI(self, list):
        probZero = len([val for val in list if val == 0]) / len(list)
        return 1 - (probZero ** 2 - (1 - probZero) ** 2)

    def splitByGini(self, X, y):
        splitBys = []
        GINIs = []
        for xCol in range(len(X[0])):
            splitBy = np.average([xRow[xCol] for xRow in X])
            splitBys.append(splitBy)
            ySplit = self.splitMatrixByAverage(X, y, xCol, splitBy)
            weightedGINI = (len(ySplit[0]) / len(y)) * self.getGINI(ySplit[0]) + (
                    len(ySplit[1]) / len(y)) * self.getGINI(ySplit[1])
            GINIs.append(weightedGINI)
        self.splitter = GINIs.index(min(GINIs))
        self.splitBy = splitBys[GINIs.index(max(GINIs))]

    def createTree(self, X, y, impurity_measure="entropy", prune=False, x_prune=None, y_prune=None):
        # if all labels are the same, set variable and return
        if (len(set(y)) <= 1):
            self.feature = y[0]

        # if all features are the same, set variable to most common label and return
        elif X.count(X[0]) == len(X):
            self.feature = max(set(y), key=y.count)

        else:
            if impurity_measure == "entropy":
                self.splitByEntropy(X, y)
            elif impurity_measure == "gini":
                self.splitByGini(X, y)

            # Create new matrices as subtrees
            child1X = []
            child1y = []
            child2X = []
            child2y = []
            for xRowIndex in range(len(X)):
                if (X[xRowIndex][self.splitter] >= self.splitBy):
                    child1X.append(X[xRowIndex])
                    child1y.append(y[xRowIndex])
                else:
                    child2X.append(X[xRowIndex])
                    child2y.append(y[xRowIndex])

            # recurse
            self.child1 = Tree()
            self.child1.createTree(child1X, child1y, impurity_measure, prune, x_prune, y_prune)
            self.child2 = Tree()
            self.child2.createTree(child2X, child2y, impurity_measure, prune, x_prune, y_prune)

            if (prune):
                self.prune(self.child1, y, x_prune, y_prune)
                self.prune(self.child2, y, x_prune, y_prune)

    def prune(self, child, y, x_prune, y_prune):
        if child.has_feature():
            return
        acc = self.accuracy(x_prune, y_prune)
        if y.count(0) > y.count(1):
            label = 0
        else:
            label = 1
        child.feature = label
        new_acc = self.accuracy(x_prune, y_prune)
        if new_acc < acc:
            child.feature = None

    def has_feature(self):
        return hasattr(self, "feature") and self.feature is not None

    def predict(self, x):
        if hasattr(self, 'feature'):
            return self.feature
        else:
            return self.child1.predict(x) if (x[self.splitter] >= self.splitBy) else self.child2.predict(x)

    def accuracy(self, x_test, y_test):
        x_result = []
        for x in x_test:
            x_result.append(self.predict(x))
        avg = 0
        for i in range(len(x_result)):
            if x_result[i] == y_test[i]:
                avg += 1
        return avg / len(x_result)


#
# X = [[3.6216, 8.6661, -2.8073, -0.44699],
#      [4.5459, 8.1674, -2.4586, -1.4621],
#      [3.866, -2.6383, 1.9242, 0.10645],
#      [3.4566, 9.5228, -4.0112, -3.5944],
#      [0.32924, -4.4552, 4.5718, -0.9888]]
# y = [1, 1, 0, 0, 0]


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
    train, prune, val, test = X[:int(len(X) * 0.6)], X[int(len(X) * 0.6): int(len(X) * 0.7)], X[int(
        len(X) * 0.7): int(len(X) * 0.8)], X[int(len(X) * 0.8):]
    return train, prune, val, test


a, b = format_data()
X, y = random_shuffle_pair(a, b)
x_train, x_prune, x_val, x_test = split(X)
y_train, y_prune, y_val, y_test = split(y)
tree1 = Tree()
tree2 = Tree()
tree1.createTree(x_train, y_train, "entropy")
tree2.createTree(x_train, y_train, "entropy", prune=True, x_prune=x_prune, y_prune=y_prune)
print(tree1.accuracy(x_test, y_test))
print(tree2.accuracy(x_test, y_test))
