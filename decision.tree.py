import numpy as np
import functools as ft


class Tree:

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
            curXCol = [xRow[xCol] for xRow in X]

            ##HER BLIR DUMT
            yAboveOrEqual = []
            yBelow = []
            for i in range(len(y)):
                if (curXCol[i] >= splitBy):
                    yAboveOrEqual.append(y[i])
                else:
                    yBelow.append(y[i])

            entAboveOrEqual = self.getEnt(yAboveOrEqual)
            entBelow = self.getEnt(yBelow)

            conditionalEnt = (len(yAboveOrEqual) / len(y)) * entAboveOrEqual + (len(yBelow) / len(y)) * entBelow;
            IGs.append(curEnt - conditionalEnt)
        # Save index of column used to split matrix, EG. column with highest information gain
        self.splitter = IGs.index(max(IGs))
        self.splitBy = splitBys[IGs.index(max(IGs))]

    def createTree(self, X, y, impurity_measure="entropy"):
        # if all labels are the same, set variable and return
        if (len(set(y)) <= 1):
            self.feature = y[0]

        # if all features are the same, set variable to most common label and return
        elif X.count(X[0]) == len(X):
            self.feature = max(set(y), key=y.count)

        else:
            if impurity_measure == "entropy":
                self.splitByEntropy(X, y)
                
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
            self.child1.createTree(child1X, child1y)
            self.child2 = Tree()
            self.child2.createTree(child2X, child2y)


    def predict(self, x):
        if hasattr(self, 'feature'):
            return self.feature
        else:
            return self.child1.predict(x) if (x[self.splitter] >= self.splitBy) else self.child2.predict(x)


X = [[3.6216, 8.6661, -2.8073, -0.44699],
     [4.5459, 8.1674, -2.4586, -1.4621],
     [3.866, -2.6383, 1.9242, 0.10645],
     [3.4566, 9.5228, -4.0112, -3.5944],
     [0.32924, -4.4552, 4.5718, -0.9888]]
y = [1, 1, 0, 0, 0]

tree1 = Tree()
tree1.createTree(X, y)
print(tree1.predict([4.6765, -3.3895, 3.4896, 1.4771]))
