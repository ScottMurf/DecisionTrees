from DecisionTrees import Tree,format_dataframe,training_test_split,accuracy,printTree
import pandas as pd
import os

#Change working directory to current directory
os.chdir(r"C:\Users\35385\Documents\GitHub\Decision Trees")
X=format_dataframe("wildfires.csv","yes")
X1,Test=training_test_split(X,0.6666)
Tree1=Tree(X1,1,6,False,"ent")
print("{}%".format(round(accuracy(Test,Tree1),2)))
printTree(Tree1)
os.chdir(r"C:\Users\35385\Documents\GitHub\Decision Trees")
