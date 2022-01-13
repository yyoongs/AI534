import pandas as pd
import numpy as np
import math
import random


class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree


def Boost_entropy(data,target_col):
    """
    This function takes target_col, which is the data column containing the class labels, and returns H(Y).

    """
    if len([x for _, x in data.groupby(data.iloc[:,target_col] >= 1)]) == 1:
        entropy = 0
        
    else:
        split_data1, split_data2 = [x for _, x in data.groupby(data.iloc[:,target_col] >= 1)]

        py0 = split_data1.iloc[:,118].sum(axis=0) / data.iloc[:,118].sum(axis=0)
        py1 = split_data2.iloc[:,118].sum(axis=0) / data.iloc[:,118].sum(axis=0)

        if py0 == 0 and py1 == 0:
            entropy = 0
        elif py0 == 0:
            entropy = -(py1*np.log2(py1))
        elif py1 == 0:
            entropy = -(py0*np.log2(py0))
        else:
            entropy = -(py0*np.log2(py0) + py1*np.log2(py1))

    return entropy


def Boost_InfoGain(data,split_attribute_name,target_name="class"):
    """
    This function calculates the information gain of specified feature. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """
    # before entropy
    before = Boost_entropy(data,117)

#     data split  df1 -> class = 1 / df2 -> class = 0
    if len([x for _, x in data.groupby(data.iloc[:,split_attribute_name] >= 1)]) == 1:
        Information_Gain = 0
        
    else:
        split_data1, split_data2 = [x for _, x in data.groupby(data.iloc[:,split_attribute_name] >= 1)]
    
        s_data1_entropy = (split_data1.iloc[:,118].sum(axis=0)/ data.iloc[:,118].sum(axis=0))*Boost_entropy(split_data1,117)
        s_data2_entropy = (split_data2.iloc[:,118].sum(axis=0)/ data.iloc[:,118].sum(axis=0))*Boost_entropy(split_data2,117)
        
        Information_Gain = before - s_data1_entropy - s_data2_entropy

    return Information_Gain


def predict(example,tree,prediction_list,default = 1):
    """
    This function handles making prediction for an example, takes two parameters:
    1. The example

    2. The tree, which is a node
    This needs to be done in a recursive manner. First check if the node is a leaf, if so, return the prediction of the node. Otherwise, send the example down the appropriate subbranch with recursive call.
    """
    if tree.feature == None:
        prediction_list.append(tree.prediction)
    elif example[tree.feature] == 0:
        predict(example,tree.left_tree,prediction_list)
    else:
        predict(example,tree.right_tree,prediction_list)
        


def Ada(data,features, depth, maxdepth, target_attribute_name="class"):
    """
    This function takes following paramters:
    1. data = the data for which the decision tree building algorithm should be run --> In the first run this equals the total dataset

    2. features = the feature space of the dataset . This is needed for the recursive call since during the tree growing process
    we have to remove features from our dataset once we have splitted on a feature

    3. target_attribute_name = the name of the target attribute
    4. depth = the current depth of the node in the tree --- this is needed to remember where you are in the overall tree
    5. maxdepth =  the stopping condition for growing the tree

    """    
    
    #First of all, define the stopping criteria here are some, depending on how you implement your code, there maybe more corner cases to consider
    """
    1. If max depth is met, return a leaf node labeled with majority class, additionally
    2. If all target_values have the same value (pure), return a leaf node labeled with majority class 
    3. If the remaining feature space is empty, return a leaf node labeled with majority class
    """
    
    leaf = True
    check = 0
    for fea in features:
        if data.iloc[:,fea].value_counts().max() < len(data):
            check = fea
            leaf = False
            break
            
    if(depth == maxdepth or leaf== True or len(features) == 0):
        predict = data.loc[:,target_attribute_name].value_counts().idxmax()
        node = Node(predict, None, None, None, None)
        return node
    
    #If none of the above holds true, grow the tree!
    #First, select the feature which best splits the dataset
    gain = {'idx': None, 'value': 0}

    for attr_idx in features:
        attr_gain = Boost_InfoGain(data, attr_idx, target_name="class")
        if attr_gain >= gain['value']:
            gain['idx'] = attr_idx
            gain['value'] = attr_gain
            
#     print("index : ",gain['idx'])
#     print("value : ",gain['value'])

    #Once best split is decided, do the following: 
    """
    1. create a node to store the selected feature 
    2. remove the selected feature from further consideration
    3. split the training data into the left and right branches and grow the left and right branch by making appropriate cursive calls
    4. return the completed node
    """
    if gain['value'] == 0:
        gain['idx'] = check

    data0, data1 = [x for _, x in data.groupby(data.iloc[:,gain['idx']] >= 1)]
    
    features.remove(gain['idx'])
    depth += 1
    
    node = Node(None,gain['idx'], None, Ada(data0,features,depth,maxdepth), Ada(data1,features,depth,maxdepth))
    
    return node



train = pd.read_csv('mushroom-train.csv')

train.loc[:,['class']] = train.loc[:,['class']].replace(0,-1)
train.insert(118, "D", 1/4500)
train.insert(119,"check",0)


validation = pd.read_csv('mushroom-val.csv')
validation.loc[:,['class']] = validation.loc[:,['class']].replace(0,-1)
validation.insert(118, "D", 1/4500)
validation.insert(119,"check",0)

feat = [_ for _ in range(117)]
alpha_list = []
tree_list = []
acc_list = []
val_acc_list = []
class_value = train['class'].values.tolist()
val_class_value = validation['class'].values.tolist()

for i in range(50):
    tree = Ada(train,feat, 0, 1, target_attribute_name="class")
    tree_list.append(tree)

    predict_list = []
    for _ in range(len(train)):
        predict(train.iloc[_,:],tree,predict_list)

    error = 0
    for i in range(len(predict_list)):
        if predict_list[i] == class_value[i]:
            train.iloc[i,119] = 1
        else:
            train.iloc[i,119] = -1
            error += train.iloc[i,118]

    # print("\n===========error==========")
    # print(error)
    if error == 0:
        error = 0.0001
        
    alpha = 1/2* math.log( (1-error) / error )
    alpha_list.append(alpha)

    train['D'] = np.where(train['check'] == 1, train['D']* np.exp(-alpha),  train['D']* np.exp(alpha))
    
    if sum(train.loc[train['check'] > 0]['D']) == 0:
        norm1 = 1
    else:
        norm1 = 0.5 / sum(train.loc[train['check'] > 0]['D'])
        
    if sum(train.loc[train['check'] < 0]['D']) == 0:
        norm2 = 1
    else:
        norm2 = 0.5 / sum(train.loc[train['check'] < 0]['D'])
        
    train['D'] = np.where(train['check'] == 1, train['D']* norm1,  train['D']* norm2)

    result = [0] * len(train)
    for j in range(len(tree_list)):

        predict_list = []
        for _ in range(len(train)):
            predict(train.iloc[_,:],tree_list[j],predict_list)

        for i in range(len(predict_list)):
            result[i] += predict_list[i] * alpha_list[j]


    result = np.sign(result)
    count = 0
    for i in range(len(result)):
        if result[i] == class_value[i]:
            count += 1

    accuracy = count / len(result)
    print("\n===========result==========")
    print(accuracy)
    acc_list.append(accuracy)
    
    
    val_result = [0] * len(validation)
    for j in range(len(tree_list)):

        predict_list = []
        for _ in range(len(validation)):
            predict(validation.iloc[_,:],tree_list[j],predict_list)

        for i in range(len(predict_list)):
            val_result[i] += predict_list[i] * alpha_list[j]


    val_result = np.sign(val_result)
    count1 = 0
    for i in range(len(val_result)):
        if val_result[i] == val_class_value[i]:
            count1 += 1

    accuracy = count1 / len(val_result)
    print("\n===========result==========")
    print(accuracy)
    val_acc_list.append(accuracy)

print(acc_list,val_acc_list)
    
graph = []
graph.append(acc_list)
graph.append(val_acc_list)

numpy_array = np.array(graph)
transpose = numpy_array.T

spar_data = pd.DataFrame(transpose,columns=['Training','Validation'])
spar_data.plot(title="AdaBoost Accuracy with dmax=5",xlabel="T",ylabel="accuracy")
# calculate accuracy