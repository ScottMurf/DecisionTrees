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
Root = Node(X1)
# Print the accuracy
print("{}%".format(round(accuracy(Test, Root), 2)))
