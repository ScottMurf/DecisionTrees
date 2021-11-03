from DecisionTrees import Tree,format_dataframe,training_test_split,accuracy,printTree
import pandas as pd

X=format_dataframe("wildfires.csv","yes")
X1,Test=training_test_split(X,0.6666)
Tree1=Tree(X1,1,6,False,"ent")
print("{}%".format(round(accuracy(Test,Tree1),2)))
printTree(Tree1)
