from implementation import Node, format_dataframe, training_test_split
from implementation import accuracy, printTree, visualise_splits, visualise_best_feature
import os
from sklearn import tree
from sklearn.metrics import accuracy_score

# Change working directory to current directory
os.chdir(r"C:\Users\35385\Documents\GitHub\Decision Trees")
# Format dataframe
X = format_dataframe("weathertxt.csv", "Play?")
# Train/Test Splitting
X1, Test = training_test_split(X, 0.6666)
# Create the Tree
Root = Node(X1,max_depth = 6)
# Print the accuracy
print("{}%".format(round(accuracy(Test, Root), 2)))

# uncomment to see Scikit-Learn Implementation on data
#clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=6)
#clf = clf.fit(X1[X1.columns[:-1]], X1["Label"])
#y_pred = clf.predict(Test[Test.columns[:-1]])
#print(accuracy_score(Test["Label"], y_pred)*100)

# uncomment to print the Tree structure
#printTree(Root)

# uncomment to print the information gain of splits for the
# temp feature on the root node
#visualise_splits(Root.df,"temp","ent")

# uncomment to visualise all splits for the root node
#visualise_best_feature(X,"ent")
