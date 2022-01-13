import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


table = pd.read_csv('IA2-train.csv')
table2 = pd.read_csv('IA2-dev.csv')

# table2
x_sum = pd.concat([table,table2])
x_sum = x_sum.reset_index(drop=True)

xi = x_sum.iloc[:1000,:197]


alpha = pd.Series(0, index=list(range(xi.shape[0])),dtype='int')

# set hyperparameter
maxiter = 100
p = 1


predict_acc_list = []
run_time = {}

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
y = x_sum.loc[:999,['Response']]
y = y.replace(0,-1)
yn = y['Response'].to_numpy()
y_result = y['Response'].values.tolist()

# for prediction
yn_predict = np.expand_dims(yn, axis=0)
yn_predict = np.repeat(yn_predict,repeats=xi.shape[0],axis=0)

# compute kernel matrix
k_matrix = np.power(np.dot(xi.to_numpy(),xi.to_numpy().T),p)

start_time = time.time()
for itr in range(maxiter):
    for i in range(xi.shape[0]):
        u = np.sum(alpha*k_matrix[i]*yn)
        if u*yn[i] <= 0:
            alpha[i] += 1
    
    # training prediction
    alpha_predict = np.repeat(np.expand_dims(alpha, axis=0),repeats=k_matrix.shape[0],axis=0)

    prediction = np.where(np.sum((alpha_predict*k_matrix*yn_predict),axis=1)>=0,1,-1)
    
    count = np.sum(np.where((prediction - y_result)==0,1,0))

    predict_acc = count / len(prediction)
    print("\n\n========", itr,"th predicted training accuracy========")
    print(predict_acc)
    predict_acc_list.append(predict_acc) 

run_time['online perceptron'] = time.time() - start_time

print(predict_acc_list)
print("Learning time:\n")
print(run_time['online perceptron'])
