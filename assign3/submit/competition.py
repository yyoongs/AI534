import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as deepcopy
import math

table = pd.read_csv('IA2-train.csv')
table2 = pd.read_csv('IA2-dev.csv')
w = pd.Series(1, index=list(range(492)), dtype='float32')

xi = pd.concat([table,table2])
xi = xi.reset_index(drop=True)

# xi = df.iloc[:,:197]

reg = 10**(-3)
learning_rate = 10**(-1)
epsilon = 10**(-5)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sign(x):
    result = pd.Series(0, index=list(range(492)), dtype='float32')
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
    result = pd.Series(0, index=list(range(492)), dtype='float32')
    i =0
    for value in w:
        if (abs(value) - reg*learning_rate) > 0:
            result[i] = abs(value) - reg*learning_rate
            i += 1
        else:
            result[i] = 0
            i += 1
            
    return result

# # normalizaion (age, annual_premium, vintage)
# mean = []
# std = []

# mean.append(xi.loc[:,"Age"].mean())
# std.append(xi.loc[:,"Age"].std())

# mean.append(xi.loc[:,"Annual_Premium"].mean())
# std.append(xi.loc[:,"Annual_Premium"].std())

# mean.append(xi.loc[:,"Vintage"].mean())
# std.append(xi.loc[:,"Vintage"].std())

xi.insert(2, "age_2", 0)
xi.insert(3, "age_3", 0)
xi.insert(4, "age_4", 0)
xi.insert(5, "age_5", 0)
xi.insert(6, "age_6", 0)
xi.insert(7, "age_7", 0)
xi.insert(8, "age_8", 0)

for i in xi.index:
    if xi.loc[i,'Age'] >= 20 and xi.loc[i,'Age'] <30:
        xi.loc[i,'age_2'] = 1
    if xi.loc[i,'Age'] >= 30 and xi.loc[i,'Age'] <40:
        xi.loc[i,'age_3'] = 1
    if xi.loc[i,'Age'] >= 40 and xi.loc[i,'Age'] <50:
        xi.loc[i,'age_4'] = 1
    if xi.loc[i,'Age'] >= 50 and xi.loc[i,'Age'] <60:
        xi.loc[i,'age_5'] = 1
    if xi.loc[i,'Age'] >= 60 and xi.loc[i,'Age'] <70:
        xi.loc[i,'age_6'] = 1
    if xi.loc[i,'Age'] >= 70 and xi.loc[i,'Age'] <80:
        xi.loc[i,'age_7'] = 1
    if xi.loc[i,'Age'] >= 80 and xi.loc[i,'Age'] <90:
        xi.loc[i,'age_8'] = 1
        
xi = xi.drop('Age',axis=1)


# xi.loc[:,"Age"] = (xi.loc[:,"Age"] - xi.loc[:,"Age"].mean()) / xi.loc[:,"Age"].std()
xi.loc[:,"Annual_Premium"] = (xi.loc[:,"Annual_Premium"] - xi.loc[:,"Annual_Premium"].mean()) / xi.loc[:,"Annual_Premium"].std()
# xi.loc[:,"Vintage"] = (xi.loc[:,"Vintage"] - xi.loc[:,"Vintage"].mean()) / xi.loc[:,"Vintage"].std()


for i in range(3):
    train=xi.sample(frac=0.8,random_state=200) #random state is a seed value
    test=xi.drop(train.index).sample(frac=1.0)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    y = train.loc[:,['Response']]
    yn = y['Response'].to_numpy()

    train = train.iloc[:,:203]

    dy = pd.get_dummies(train.Vintage, prefix='Vintage')
    train = pd.concat([train,dy],axis=1)
    train = train.drop('Vintage',axis=1)

    # # # print(w)
    for i in range(10000):
        print("==========Epoch ",i," ===============")
        result = yn - sigmoid(np.sum(train.to_numpy() * w.to_numpy(),axis=1))
        result1 = np.expand_dims(result, axis=0)
        result1 = np.repeat(result1,repeats=492,axis=0)

        xr = np.multiply(train,result1.T)
    #     print(xr.to_numpy())
        w += learning_rate*(np.sum(xr,axis=0)/12800).to_numpy()
    #     print(w)
        # L2 regularization
        # w[1:] -= reg*learning_rate*w[1:]

        # L1 regularization
        sign_w = sign(w)
        w[1:] = sign_w[1:]*max_w(w,reg,learning_rate)[1:]

        dw = max(abs((np.sum(xr,axis=0)/12800).to_numpy()))

        if dw < 10**(-5):
            print("break is worked")
            break
    # #     print(abs(max(abs((np.sum(xr,axis=0)/6000).to_numpy())) - dw))
    #     if abs(max(abs((np.sum(xr,axis=0)/12800).to_numpy())) - dw) < 10**(-7):
    #         print("break is worked")
    #         break
    #     dw = max(abs((np.sum(xr,axis=0)/12800).to_numpy()))
        print(dw)


#     print(reg*learning_rate*w[1:])
print(w.tolist())

print("\nlearning rate : ",learning_rate, "\nreg : ", reg, "\nepsilon : ", epsilon)


data = train.to_numpy() * w.to_numpy()
y_result = y['Response'].values.tolist()



# print(np.sum(xr.iloc[:,1:],axis=1) - xr.iloc[:,0])

predict = []
i = 0
for _ in np.sum(data,axis=1):
    if _ > 0:
        predict.append(1)
    else:
        predict.append(0)
    i += 1
    
# print(predict)
# print(y_result)
count = 0
for i in range(12800):
    if predict[i] == y_result[i]:
        count += 1

predict_acc = count / 12800
print("\n\n========predicted training accuracy========")
print(predict_acc)   

# =======================================================
# =======================================================
# =======================================================
# =======================================================

y = test.loc[:,['Response']]
# yn = y['Response'].to_numpy()

test = test.iloc[:,:203]

dy = pd.get_dummies(test.Vintage, prefix='Vintage')
test = pd.concat([test,dy],axis=1)
test = test.drop('Vintage',axis=1)

data = test.to_numpy() * w.to_numpy()
y_result = y['Response'].values.tolist()

# print("\n\n")
# print(w.tolist())

predict = []
i = 0
for _ in np.sum(data,axis=1):
    if _ > 0:
        predict.append(1)
    else:
        predict.append(0)
    i += 1
    
# print(predict)
# print(y_result)
count = 0
for i in range(3200):
    if predict[i] == y_result[i]:
        count += 1

predict_acc = count / 3200
print("\n\n========predicted validation accuracy========")
print(predict_acc) 