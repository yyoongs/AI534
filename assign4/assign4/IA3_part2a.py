import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time

table = pd.read_csv('IA2-train.csv')

# initialize alpha
alpha = pd.Series(0, index=list(range(6000)),dtype='int')
xi = table.iloc[:,:197]

# set hyperparameter
maxiter = 100
p = 1


predict_acc_list = []
predict_val_acc_list = []

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

# for prediction
yn_predict = np.expand_dims(yn, axis=0)
yn_predict = np.repeat(yn_predict,repeats=xi.shape[0],axis=0)


# validation dataset preprocessing
table2 = pd.read_csv('IA2-dev.csv')
val_xi = table2.iloc[:,:197]

val_xi.loc[:,"Age"] = (val_xi.loc[:,"Age"] - mean.pop(0)) / std.pop(0)
val_xi.loc[:,"Annual_Premium"] = (val_xi.loc[:,"Annual_Premium"] - mean.pop(0)) / std.pop(0)
val_xi.loc[:,"Vintage"] = (val_xi.loc[:,"Vintage"] - mean.pop(0)) / std.pop(0)

# result yn
val_y = table2.loc[:,['Response']]
val_y = val_y.replace(0,-1)
val_yn = val_y['Response'].to_numpy()
val_y_result = val_y['Response'].values.tolist()


# compute kernel matrix
k_matrix = np.power(np.dot(xi.to_numpy(),xi.to_numpy().T),p)
val_matrix = np.power(np.dot(val_xi.to_numpy(),xi.to_numpy().T),p)

val_yn_predict = np.expand_dims(yn, axis=0)
val_yn_predict = np.repeat(val_yn_predict,repeats=val_xi.shape[0],axis=0)

for itr in range(maxiter):
    for i in range(xi.shape[0]):
        u = np.sum(alpha*k_matrix[i]*yn)
        if u*yn[i] <= 0:
            alpha[i] += 1
    
    alpha_predict = np.expand_dims(alpha, axis=0)
    alpha_predict = np.repeat(alpha_predict,repeats=k_matrix.shape[0],axis=0)
    
    # training prediction
    prediction = np.sum((alpha_predict*k_matrix*yn_predict),axis=1)
    for i in range(len(prediction)):
        if prediction[i] >= 0:
            prediction[i] = 1
        else:
            prediction[i] = -1

    count = 0
    for i in range(len(prediction)):
        if prediction[i] == y_result[i]:
            count += 1

    predict_acc = count / len(prediction)
    print("\n\n========", itr,"th predicted training accuracy========")
    print(predict_acc)
    predict_acc_list.append(predict_acc) 
    
    # validation prediction
    val_alpha_predict = np.expand_dims(alpha, axis=0)
    val_alpha_predict = np.repeat(val_alpha_predict,repeats=val_matrix.shape[0],axis=0)

    val_prediction = np.sum((val_alpha_predict*val_matrix*val_yn_predict),axis=1)
    for i in range(len(val_prediction)):
        if val_prediction[i] >= 0:
            val_prediction[i] = 1
        else:
            val_prediction[i] = -1

    count1 = 0
    for i in range(len(val_prediction)):
        if val_prediction[i] == val_y_result[i]:
            count1 += 1

    val_predict_acc = count1 / len(val_prediction)
    print("\n\n========", itr,"th predicted validation accuracy========")
    print(val_predict_acc)
    predict_val_acc_list.append(val_predict_acc) 

print(predict_acc_list)
print(predict_val_acc_list)
