from DecisionTrees import Node, format_dataframe, training_test_split
from DecisionTrees import accuracy, printTree
import os
from sklearn import tree
from sklearn.metrics import accuracy_score

# Change working directory to current directory
os.chdir(r"C:\Users\35385\Documents\GitHub\Decision Trees")
# Format dataframeaccuracy_score(Test_labs, y_pred)*100
X = format_dataframe("wildfires.csv", "yes")
# Train/Test Splitting
X1, Test = training_test_split(X, 0.6666)
# Create the Tree
Root = Node(X1, 1, 6, False, "ent")
# Print the accuracy
print("{}%".format(round(accuracy(Test, Root), 2)))

#Scikit-Learn Implementation
#clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=6)
#clf = clf.fit(X1[X1.columns[:-1]], X1["Label"])
#y_pred = clf.predict(Test[Test.columns[:-1]])
#print(accuracy_score(Test["Label"], y_pred)*100)

# Print the Tree structure
printTree(Root)
