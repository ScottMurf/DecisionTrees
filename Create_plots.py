


from DecisionTrees import training_test_split, format_dataframe
from DecisionTrees import accuracy, Node
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np
import os
# Change working directory to current directory
os.chdir(r"C:\Users\35385\Documents\GitHub\Decision Trees")
# Format dataframe
X = format_dataframe("wildfires.csv", "yes")
#list to store accuracy of custom C4.5 algorithm
accs = []
#list to store accuracy of scikit learn algorithm
sciaccs = []
N = 10
#Get accuracy of both algorithms 10 times and store the accuracies
for i in range(N):
    X1, Test = training_test_split(X, 0.6666)
    Root = Node(X1, 1, 100, False, "gini")
    Train = X1.drop("Label", axis=1)
    Train_labs = X1["Label"]
    Test1 = Test.drop("Label", axis=1)
    Test_labs = Test["Label"]
    clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=100)
    clf = clf.fit(Train, Train_labs)
    y_pred = clf.predict(Test1)

    sciaccs.append(accuracy_score(Test_labs, y_pred)*100)
    accs.append(accuracy(Test, Root))

#Get averages of the accuracies
average1 = sum(accs)/len(accs)
average2 = sum(sciaccs)/len(sciaccs)
#indices
ind = np.arange(N)
#bin width
width = 0.3
#Create bar charts for the accuracies of the algorithms
plt.figure(figsize=[20, 20])
plt.bar(ind, tuple(accs), width, label=("Average Accuracy={}%".format(round(average1, 2))))
plt.bar(ind+width, tuple(sciaccs), width, label=("Scikit Learn average accuracy={}%".format(round(average2, 2))))
plt.ylim(0, 100)
plt.title("Accuracy of Decision Tree Algorithm on 10 iterations of randomized training and test data. criterion=gini",fontsize=10)
plt.ylabel("Accuracy (%)")
plt.xlabel("Iteration")
plt.xticks(ind + width / 2, ('Iteration 1','Iteration 2','Iteration 3','Iteration 4','Iteration 5','Iteration 6','Iteration 7','Iteration 8','Iteration 9','Iteration 10',))
plt.legend(loc=2, prop={'size': 10})
plt.show()
