import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as deepcopy
import math

table = pd.read_csv('IA2-train.csv')
w = pd.Series(1, index=list(range(197)), dtype='float32')

xi = table.iloc[:,:197]

reg = 10**(-3)
learning_rate = 10**(-1)
epsilon = 10**(-5)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sign(x):
    result = pd.Series(0, index=list(range(197)), dtype='float32')
    i =0
    for value in x:
        if value > 0:
            result[i] = 1
            i += 1
        elif value == 0:
            result[i] = 0
            i += 1 
        else:
            result[i] = -1
            i += 1
    return result
            
def max_w(w,reg,learning_rate):
    result = pd.Series(0, index=list(range(197)), dtype='float32')
    i =0
    for value in w:
        if (abs(value) - reg*learning_rate) > 0:
            result[i] = abs(value) - reg*learning_rate
            i += 1
        else:
            result[i] = 0
            i += 1
            
    return result


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
yn = y['Response'].to_numpy()

# iteration
for i in range(10000):
    print("==========Epoch ",i," ===============")
    result = yn - sigmoid(np.sum(xi.to_numpy() * w.to_numpy(),axis=1))
    result1 = np.expand_dims(result, axis=0)
    result1 = np.repeat(result1,repeats=197,axis=0)

    xr = np.multiply(xi,result1.T)
    w += learning_rate*(np.sum(xr,axis=0)/6000).to_numpy()

    # L1 regulizaion
    sign_w = sign(w)
    w[1:] = sign_w[1:]*max_w(w,reg,learning_rate)[1:]

    dw = max(abs((np.sum(xr,axis=0)/6000).to_numpy()))

    if dw < 10**(-5):
        print("break is worked")
        break

    print(dw)
    
print(w.tolist())

print("\nlearning rate : ",learning_rate, "\nreg : ", reg, "\nepsilon : ", epsilon)


# calculate accuracy
data = xi.to_numpy() * w.to_numpy()
y_result = y['Response'].values.tolist()

predict = []
i = 0
for _ in np.sum(data,axis=1):
    if _ > 0:
        predict.append(1)
    else:
        predict.append(0)
    i += 1

count = 0
for i in range(6000):
    if predict[i] == y_result[i]:
        count += 1

predict_acc = count / 6000
print("\n\n========predicted training accuracy========")
print(predict_acc)   

# =============================================================
# calculate validation accuracy
table = pd.read_csv('IA2-dev.csv')

xi = table.iloc[:,:197]

xi.loc[:,"Age"] = (xi.loc[:,"Age"] - mean.pop(0)) / std.pop(0)
xi.loc[:,"Annual_Premium"] = (xi.loc[:,"Annual_Premium"] - mean.pop(0)) / std.pop(0)
xi.loc[:,"Vintage"] = (xi.loc[:,"Vintage"] - mean.pop(0)) / std.pop(0)

# result y
y = table.loc[:,['Response']]
yn = y['Response'].to_numpy()

data = xi.to_numpy() * w.to_numpy()
y_result = y['Response'].values.tolist()

predict = []
i = 0
for _ in np.sum(data,axis=1):
    if _ > 0:
        predict.append(1)
    else:
        predict.append(0)
    i += 1

count = 0
for i in range(10000):
    if predict[i] == y_result[i]:
        count += 1

predict_acc = count / 10000
print("\n\n========predicted validation accuracy========")
print(predict_acc) 