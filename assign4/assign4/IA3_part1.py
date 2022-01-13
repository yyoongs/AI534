import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


table = pd.read_csv('IA2-train.csv')

# initialize w and avg_w
w = pd.Series(1, index=list(range(197)), dtype='float32')
avg_w = pd.Series(1, index=list(range(197)), dtype='float32')

xi = table.iloc[:,:197]

# set hyperparameter
maxiter = 100
s = 1

# accuracy of training with w
predict_acc_list = []

# accuracy of training with avg_w
predict_avg_acc_list = []

# accuracy of validation with w
predict_val_acc_list = []

# accuracy of validation with avg_w
predict_val_avg_acc_list = []

# normalizaion (age, annual_premium, vintage)
mean = []
std = []

mean.append(xi.loc[:,"Age"].mean())
std.append(xi.loc[:,"Age"].std())

mean.append(xi.loc[:,"Annual_Premium"].mean())
std.append(xi.loc[:,"Annual_Premium"].std())

mean.append(xi.loc[:,"Vintage"].mean())
std.append(xi.loc[:,"Vintage"].std())

xi.loc[:,"Age"] = (xi.loc[:,"Age"] - xi.loc[:,"Age"].mean()) / xi.loc[:,"Age"].std()
xi.loc[:,"Annual_Premium"] = (xi.loc[:,"Annual_Premium"] - xi.loc[:,"Annual_Premium"].mean()) / xi.loc[:,"Annual_Premium"].std()
xi.loc[:,"Vintage"] = (xi.loc[:,"Vintage"] - xi.loc[:,"Vintage"].mean()) / xi.loc[:,"Vintage"].std()

# result y
y = table.loc[:,['Response']]
y = y.replace(0,-1)
yn = y['Response'].to_numpy()
y_result = y['Response'].values.tolist()


# validation dataset preprocessing
table2 = pd.read_csv('IA2-dev.csv')

val_xi = table2.iloc[:,:197]

val_xi.loc[:,"Age"] = (val_xi.loc[:,"Age"] - mean.pop(0)) / std.pop(0)
val_xi.loc[:,"Annual_Premium"] = (val_xi.loc[:,"Annual_Premium"] - mean.pop(0)) / std.pop(0)
val_xi.loc[:,"Vintage"] = (val_xi.loc[:,"Vintage"] - mean.pop(0)) / std.pop(0)

# result val_y
val_y = table2.loc[:,['Response']]
val_y = val_y.replace(0,-1)
val_yn = val_y['Response'].to_numpy()
val_y_result = val_y['Response'].values.tolist()

for itr in range(maxiter):
    for ex in range(xi.shape[0]):
        if yn[ex] * np.sum(xi.iloc[ex,:].to_numpy() * w.to_numpy()) <= 0:
            w += yn[ex]*xi.iloc[ex,:].to_numpy()
        avg_w = (s*avg_w + w) / (s+1)
        s += 1
    
    # calculate accuracy of training with w
    data = xi.to_numpy() * w.to_numpy()
    
    predict_w = []
    for _ in np.sum(data,axis=1):
        if _ >= 0:
            predict_w.append(1)
        else:
            predict_w.append(-1)

    count = 0
    for i in range(xi.shape[0]):
        if predict_w[i] == y_result[i]:
            count += 1

    predict_acc = count / xi.shape[0]


    # calculate accuracy of training with avg_w

    data2 = xi.to_numpy() * avg_w.to_numpy()

    predict_avg_w = []
    for _ in np.sum(data2,axis=1):
        if _ >= 0:
            predict_avg_w.append(1)
        else:
            predict_avg_w.append(-1)

    count1 = 0
    for i in range(xi.shape[0]):
        if predict_avg_w[i] == y_result[i]:
            count1 += 1

    predict_avg_acc = count1 / xi.shape[0]

    print("\n\n========", itr,"th predicted training accuracy========")
    print("w perceptron acc: ",predict_acc)   
    print("w avg perceptron acc: ",predict_avg_acc)
    predict_acc_list.append(predict_acc) 
    predict_avg_acc_list.append(predict_avg_acc)

    # validation
    data3 = val_xi.to_numpy() * w.to_numpy()

    predict_val_w = []
    i = 0
    for _ in np.sum(data3,axis=1):
        if _ > 0:
            predict_val_w.append(1)
        else:
            predict_val_w.append(-1)
        i += 1

    count3 = 0
    for i in range(val_xi.shape[0]):
        if predict_val_w[i] == val_y_result[i]:
            count3 += 1

    predict_val_acc = count3 / val_xi.shape[0]

    # calculate accuracy of validation with avg_w
    data4 = val_xi.to_numpy() * avg_w.to_numpy()
    
    predict_val_avg_w = []
    for _ in np.sum(data4,axis=1):
        if _ >= 0:
            predict_val_avg_w.append(1)
        else:
            predict_val_avg_w.append(-1)

    count4 = 0
    for i in range(val_xi.shape[0]):
        if predict_val_avg_w[i] == val_y_result[i]:
            count4 += 1

    predict_val_avg_acc = count4 / val_xi.shape[0]


    print("\n\n========predicted validation accuracy========")
    print(predict_val_acc) 
    print(predict_val_avg_acc)
    predict_val_acc_list.append(predict_val_acc) 
    predict_val_avg_acc_list.append(predict_val_avg_acc)

    
    

print("predict_acc_list",predict_acc_list)
print("predict_avg_acc_list",predict_avg_acc_list)
print("predict_val_acc_list",predict_val_acc_list)
print("predict_val_avg_acc_list",predict_val_avg_acc_list)
