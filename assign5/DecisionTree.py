import pandas as pd
import numpy as np
import math



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



def entropy(target_col):
    """
    This function takes target_col, which is the data column containing the class labels, and returns H(Y).

    """
    py0 = len(np.where(target_col==0)[0])/len(target_col)
    py1 = len(np.where(target_col==1)[0])/len(target_col)
    
    if py0 == 0 and py1 == 0:
        entropy = 0
    elif py0 == 0:
        entropy = -(py1*np.log2(py1))
    elif py1 == 0:
        entropy = -(py0*np.log2(py0))
    else:
        entropy = -(py0*np.log2(py0) + py1*np.log2(py1))

    return entropy


def InfoGain(data,split_attribute_name,target_name="class"):
    """
    This function calculates the information gain of specified feature. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """
    # before entropy
    before = entropy(data[target_name].to_numpy())

#     data split  df1 -> class = 1 / df2 -> class = 0
    if len([x for _, x in data.groupby(data.iloc[:,split_attribute_name] >= 1)]) == 1:
        Information_Gain = 0
        
    else:
        split_data1, split_data2 = [x for _, x in data.groupby(data.iloc[:,split_attribute_name] >= 1)]
    
        s_data1_entropy = (len(split_data1)/len(data))*entropy(split_data1[target_name].to_numpy())
        s_data2_entropy = (len(split_data2)/len(data))*entropy(split_data2[target_name].to_numpy())

        Information_Gain = before - s_data1_entropy - s_data2_entropy

    return Information_Gain


def DecisionTree(data,features, depth, maxdepth, target_attribute_name="class"):
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
        attr_gain = InfoGain(data, attr_idx, target_name="class")
        if attr_gain >= gain['value']:
            gain['idx'] = attr_idx
            gain['value'] = attr_gain
            
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
    print("selected feature : ",gain['idx'])
    print("Information gain : ",gain['value'])

    node = Node(None,gain['idx'], None, DecisionTree(data0,features,depth,maxdepth), DecisionTree(data1,features,depth,maxdepth))
    
    return node


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
        




train = pd.read_csv('mushroom-train.csv')
validation = pd.read_csv('mushroom-val.csv')


feat = [_ for _ in range(117)]

# set dmax here
tree = DecisionTree(train,feat, 0, 2, target_attribute_name="class")

predict_list = []
acc_list = []
val_list = []
for _ in range(len(train)):
    predict(train.iloc[_,:],tree,predict_list)

class_value = train['class'].values.tolist()

count1 = 0
for i in range(len(predict_list)):
    if predict_list[i] == class_value[i]:
        count1 += 1

predict_acc = count1 / len(predict_list)
print("\n===========training acc result==========")
print(predict_acc)
acc_list.append(predict_acc)

predict_list = []
for _ in range(len(validation)):
    predict(validation.iloc[_,:],tree,predict_list)

class_value = validation['class'].values.tolist()

count = 0
for i in range(len(predict_list)):
    if predict_list[i] == class_value[i]:
        count += 1

val_acc = count / len(predict_list)
print("\n===========validation acc result==========")
print(val_acc)
val_list.append(val_acc)



graph = [[0.8857777777777778,0.93,0.9586666666666667,0.9842222222222222,1.0,1.0,1.0,0.9986666666666667,1.0,1.0],[0.89125,0.916875,0.968125,0.991875,1.0,1.0,1.0,0.99875,1.0,1.0]]

numpy_array = np.array(graph)
transpose = numpy_array.T

spar_data = pd.DataFrame(transpose,columns=['Training','Validation'])
spar_data.plot(title="Accuracy of decision tree",xlabel="dmax",ylabel="accuracy")
