from DecisionTrees import Tree, format_dataframe, training_test_split
from DecisionTrees import accuracy, printTree
import os

# Change working directory to current directory
os.chdir(r"C:\Users\35385\Documents\GitHub\Decision Trees")
# Format dataframe
X = format_dataframe("wildfires.csv", "yes")
# Train/Test Splitting
X1, Test = training_test_split(X, 0.6666)
# Create the Tree
Tree1 = Tree(X1, 1, 6, False, "ent")
# Print the accuracy
print("{}%".format(round(accuracy(Test, Tree1), 2)))
# Print the Tree structure
printTree(Tree1)
