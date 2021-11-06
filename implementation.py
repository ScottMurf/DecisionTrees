import pandas as pd
import numpy as np
import csv
from collections import Counter
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def format_dataframe(csv,Label):
    """formats dataframe to fit other function needs"""
    Df=pd.read_csv(csv)
    Label_list=Df[Label].to_frame()
    new_cols=list(Df.columns)
    new_cols.remove(Label)
    X=Df[new_cols]
    X["Label"]=Label_list
    return X

def training_test_split(Df,prop):
    """Divides dataframes into training and test dataframes"""
    Df=Df.sample(frac=1).reset_index().drop("index",axis=1)
    length1=int(prop*len(Df))
    Training=Df.loc[:length1,]
    Test=Df.loc[length1:,].reset_index().drop("index",axis=1)

    return Training,Test

def get_vals(X):
    """Function to return all unique
    values for each feature in a dataframe"""

    vals_dict={}
    #For each feature in the dataset
    for i in X.columns:
        temp_list=[]
        for x in range(len(X)):
            temp_list.append(X[i][x])
        #Add this feature and its unique values to dictionary
        vals_dict[i]=set(temp_list)

    return vals_dict

def splits(feat,X):
    """Takes a feature and returns all the possible
    ways to split for the data using the midpoint of all the feature values"""

    #Get the sorted list of values for a feature
    vals=sorted(list(get_vals(X)[feat]))

    #Create list of the midpoint values for all adjacent values
    splits=[]
    for i in range(len(vals)-1):
        splits.append((vals[i]+vals[i+1])/2)

    return splits

def splitter(X,feat,split):
    """This function splits the dataset above and below
    a certain feature threshold"""
    #Lists for values above and below split value
    under=[]
    over=[]
    #Append lists accordingly
    for i in range(len(X)):
        if X[feat][i]<split:
            under.append(i)
        else:
            over.append(i)
    return(over,under)

def ent(l1,l2):
    """Calculates the entropy when 2
    classes are present in the data"""
    #Special cas
    if l1==0 and l2==0:
        return 0
    else:
        #Get proprotions
        p1=l1/(l1+l2)
        p2=l2/(l1+l2)
        #Special cases
        if p1==0:
            return -(p2)*np.log2(p2)
        if p2==0:
            return -(p1)*np.log2(p1)
        if p2==0 and p1==0:
            return 0
        #Normal case
        else:
            return -(p1)*np.log2(p1)-(p2)*np.log2(p2)

def gini(l1,l2):
    """Calculates the gini index when 2
    classes are present in the data"""
    #Special case
    if l1==0 and l2==0:
        return 0
    #Get proportions
    p1=l1/(l1+l2)
    p2=l2/(l1+l2)

    return 1-(p1)**2-(p2)**2

def numerical_info(X,feat,split):
    """Function to get the information gain of a specific
    feature split in the data for a numerical feature"""

    #Get the labels and lists of labels
    labels=list(set(X["Label"]))
    label_list=X["Label"]
    #Split the data
    over_indices,under_indices=splitter(X,feat,split)
    #Get labels for values in under and over splits
    over_label,under_label=[],[]
    for i in range(len(label_list)):
        if i in over_indices:
            over_label.append(label_list[i])
        else:
            under_label.append(label_list[i])
    #create tuples of indices and labels for under and over splits
    under_tot=list(zip(under_indices,under_label))
    over_tot=list(zip(over_indices,over_label))

    #Create dictionaries with the counters of each label
    #for the over and under splits
    dictover= dict(Counter(i[1] for i in over_tot))
    dictunder= dict(Counter(i[1] for i in under_tot))
    for label in labels:
        if label not in dictover.keys():
            dictover[label]=0
        if label not in dictunder.keys():
            dictunder[label]=0

    #Get biinary labels splits
    o1,o2=tuple(dictover.values())
    u1,u2=tuple(dictunder.values())
    #Get entropy and gini for each split
    En1=(len(over_indices)/len(X))*ent(o1,o2)
    En2=(len(under_indices)/len(X))*ent(u1,u2)
    G1=(len(over_indices)/len(X))*gini(o1,o2)
    G2=(len(under_indices)/len(X))*gini(u1,u2)

    #Get counts of each label in unsplitted data
    dictall={}
    for label in labels:
        dictall[label]=0
    #Count label counts for unsplit data
    for label in labels:
        for lab in label_list:
            if label==lab:
                dictall[label]+=1
    all1,all2=tuple(dictall.values())
    #Get entropy and gini of unsplit data
    Ent_S=ent(all1,all2)
    Gini_S=gini(all1,all2)
    Gain_G=Gini_S-G1-G2
    Gain_E=Ent_S-En1-En2
    #return the information gain of over and under splits
    return ((Gain_E,Gain_G),dictunder,dictover)

def categorical_info(X,feat):
    """Function to get the information gain of a categorical
    feature data split."""

    #Get the labels and lists of labels
    labels=list(set(X["Label"]))
    label_list=X["Label"]
    #Get all unique feature entries
    vals=set(X[feat])
    #create a dictionary for the indices of said entries and their labels
    dict_indices={}
    for val in vals:
        dict_indices[val]=[]
    for i in range(len(X)):
        for val in vals:
            if X[feat][i]==val:
                dict_indices[val].append((i,label_list[i]))
    #Get counts of labels for each entry
    dict_tot={}
    for key in dict_indices:
        dict_tot[key]=list(dict(Counter(i[1] for i in dict_indices[key])).values())
        if len(dict_tot[key])==1:
            dict_tot[key].append(0)
    #Get counts of each label in unsplitted data
    dictall={}
    for label in labels:
        dictall[label]=0
    #Count label counts for unsplit data
    for label in labels:
        for lab in label_list:
            if label==lab:
                dictall[label]+=1
    all1,all2=tuple(dictall.values())
    #Get entropy and gini of unsplit data
    Ent_S=ent(all1,all2)
    Gini_S=gini(all1,all2)
    #Subtract Entropy/Gini of split data to get information gain
    for val in vals:
        counts=dict_tot[val]
        Ent_S-=((counts[0]+counts[1])/len(X))*ent(counts[0],counts[1])
        Gini_S-=((counts[0]+counts[1])/len(X))*gini(counts[0],counts[1])

    return (Ent_S,Gini_S),dict_tot

def best_split(X,feat,criterion):
    """Function that finds best split for a numerical feature
    to maximise information gain."""

    #Get possible splits
    splits1=splits(feat,X)
    dict1={}

    #Get entropy/gini for each split
    for split in splits1:
        if criterion=="ent":
            dict1[split]=numerical_info(X,feat,split)[0][0]
        if criterion=="gini":
            dict1[split]=numerical_info(X,feat,split)[0][1]
    #find maximum information gain
    if len(dict1)>0:
        maxi=(max(dict1, key=dict1.get))
        return(feat,maxi,dict1[maxi])
    else:
        return(feat,0,0)

def best_feature(X,criterion):

    """Function that finds the best feature by which to
    split the data X using a speicifed criterion"""

    import numpy
    Label=X["Label"]
    features=X.columns[:-1]
    feat_list=[]
    #Test all features
    for feat in features:
        #If the feature is categorical
        if type(X[feat].values[0])==str or type(X[feat].values[0])==numpy.bool_:
            if criterion=="ent":
                feat_list.append((feat,None,categorical_info(X,feat)[0][0]))
            if criterion=="gini":
                feat_list.append((feat,None,categorical_info(X,feat)[0][1]))
        #numerical features
        else:
            feat_list.append(best_split(X,feat,criterion))

    #Get feature with maximum information gain
    best_feature=sorted(feat_list,key=lambda x:x[2],reverse=True)[0]
    #If the best feature is categorical
    if type(X[best_feature[0]][1])==str or type(X[best_feature[0]][1])==numpy.bool_:
        split_info=categorical_info(X,best_feature[0])
        entries=list(split_info[1].keys())
        dict_i={}
        #Initlialise entrys and labels
        for entry in entries:
            dict_i[entry]=([],{})
            for label in set(Label):
                dict_i[entry][1][label]=0
        #Indices of each entry
        for i in range(len(X)):
            for entry in entries:
                if X[best_feature[0]][i]==entry:
                    dict_i[entry][0].append(i)
        #Label counter
        for key in dict_i.keys():
            for x in dict_i[key][0]:
                for label in dict_i[key][1].keys():
                    if label==X["Label"][x]:
                        dict_i[key][1][label]+=1
        return best_feature,dict_i

    #if the best feature is numerical
    else:
        split_info=numerical_info(X,best_feature[0],best_feature[1])
        under_info=split_info[1]
        over_info=split_info[2]
        #Overall dictionary
        dict1={}
        #keys for over and under split
        dict1["over"]=([],{})
        dict1["under"]=([],{})
        for label in set(Label):
            dict1["over"][1][label]=0
            dict1["under"][1][label]=0
        #Indices of each entry
        for i in range(len(X)):
            if X[best_feature[0]][i]<=best_feature[1]:
                dict1["under"][0].append(i)
            else:
                dict1["over"][0].append(i)
        for key in dict1.keys():
            for x in dict1[key][0]:
                for label in dict1[key][1].keys():
                    if label==X["Label"][x]:
                        dict1[key][1][label]+=1

        return best_feature,dict1

def segment(X,criterion):

    #If only one class remains, return original dataframe
    if len(X["Label"].unique())==1:
        return(X)

    else:
        #get information for the best split
        bf,dict1=best_feature(X,criterion)
        dict2={}
        #for all key(splits) in the dictionary
        for key in dict1.keys():
            #Copy the dataframe to each key
            dict2[key]=X
            #Get difference of overall dataframe and
            #indices of split dataframe
            A=set(i for i in range(len(X)))
            B=set(dict1[key][0])
            difference=A.difference(B)
            #Drop these values for that key
            for i in difference:
                dict2[key]=dict2[key].drop(i)
        #Reset indices
        for key in dict2.keys():
            dict2[key]=dict2[key].reset_index(drop=True)
    #Return dictionary containing split dataframes
    return(bf,dict2)


class Node:
    """Tree Class for creating decision tree structures"""
    def __init__(self,df,depth=1,max_depth=100,end=False,criterion="ent",key=None):

        # Attributes of a node in tree
        self.criterion=criterion
        self.depth=depth
        self.df=df
        self.end=end
        self.max_depth=max_depth
        # Temporary variable
        self.temp=None
        self.key=key

        # If max depth is reached, make Node a leaf
        if self.depth==self.max_depth:
            self.end=True

        # If not on Leaf Node
        if self.end==False:

            # Split the Node according to the best split in dataset
            self.bf,self.temp=segment(self.df,self.criterion)
            self.feat,self.threshold=self.bf[0],self.bf[1]

            if self.depth!=self.max_depth:

                # Fill the current Tree.under attribute with Tree structure
                # underneath that node
                self.under=[]
                for key in self.temp.keys():

                    if len(self.temp[key]["Label"].unique())>1:
                        self.under.append(Node(self.temp[key],self.depth+1,self.max_depth,False,self.criterion,key))

                    if len(self.temp[key]["Label"].unique())==1:
                        self.under.append(Node(self.temp[key],self.depth+1,self.max_depth,True,self.criterion,key))

    #Returns the best feature available for at specific node
    def best_feat(self):
        if self.end!=True:
            return self.bf
        else:
            return 0,0,0


def testing(observation,Node):
    """Gives back the predicted label for a specific observation"""
    # Traverse tree until reaching a leaf
    while Node.end==False:
        feat,threshold= Node.feat,Node.threshold
        # for categorical feature splits
        if Node.threshold==None:
            for i in range(len(Node.under)):
                if observation[feat].to_string(index=False)==str(Node.under[i].key):
                    Node=Node.under[i]
                    break
                else:
                    Node.end=True
        # for numerical feature splits
        else:
            if float(observation[feat])<threshold:
                Node=Node.under[1]
            elif float(observation[feat])>=threshold:
                Node=Node.under[0]

    return Counter(Node.df["Label"]).most_common()[0][0]

def predict(test,Node):
    """stores predictions for a test dataset"""
    test_labels=[]

    for i in range(len(test)):
        obs=test.loc[[i]]
        test_labels.append(testing(obs,Node))

    return test_labels

def accuracy(test,Node):
    """returns accuracy of predictions for a test set"""
    predictions=predict(test,Node)
    labels=test["Label"]
    count=0
    pred_act=list(zip(predictions,labels))

    for i in range(len(labels)):
        if (labels[i]==predictions[i]):
            count+=1
    #writes the prediction and actual labels to a text file
    f = open("predictions_and_real_labels.txt", "w")
    for i in pred_act:
        f.write("{}\n".format(i))
    f.close()

    return (count/len(labels)*100)

def printTree(Node,val="Root"):
    """Prints a text representation of tree"""

    a,b,c=Node.best_feat()
    # leaf nodes
    if a==0 and b==0 and c==0:
        print(Node.depth-1,"-"*10*(Node.depth-1),">",val,"/ Maj Class=",Counter(Node.df["Label"]).most_common(1)[0][0])
    # all other nodes
    else:
        print(Node.depth-1,"-"*10*(Node.depth-1),">",val,"/","feature:",a,"/","split:",b,"/ Maj Class=",Counter(Node.df["Label"]).most_common(1)[0][0])
    if Node.end==False:
        # recursive calls
        for node in Node.under:
            printTree(node,node.key)

#Returns the best split for a feature
def visualise_splits(X,feat,criterion):
    import matplotlib.pyplot as plt
    splits1=splits(feat,X)
    dict1={}

    for split in splits1:
        dict1[split]=0

    for split in splits1:
        if criterion=="ent":
            dict1[split]=numerical_info(X,feat,split)[0][0]
        if criterion=="gini":
            dict1[split]=numerical_info(X,feat,split)[0][1]

    y=dict1.keys()
    x=dict1.values()

    maxi=max(list(zip(x,y)))
    plt.figure(figsize=(20,5))
    plt.bar(y,x,color="b")
    plt.bar(maxi[1],maxi[0],color="r")
    plt.title("Feature: {}".format(feat))
    plt.ylabel("{}".format(criterion))
    plt.xlabel("Split value")
    plt.show()
    return(maxi)

def visualise_best_feature(X,criterion):
    import numpy
    feats=list(X.columns[:-1])
    maxis=[]
    for i in range(len(feats)):
        if type(X[feats[i]].values[0])==str or type(X[feats[i]].values[0])==numpy.bool_:
            maxis.append(categorical_info(X,feat)[0][0])
        else:
            maxis.append((visualise_splits(X,feats[i],criterion),feats[i]))

    plt.figure(figsize=(20,5))
    for val,feat in maxis:
        plt.title("Comparison of best splits for each feature")
        plt.bar(feat,val[0],color="b")
        plt.bar(max(maxis)[1],max(maxis)[0][0],color="r")
    plt.show()
    print(maxis)
