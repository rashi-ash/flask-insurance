import pandas as pd
from matplotlib import pyplot as plt
import numpy as np 
df=pd.read_csv("insurance_data.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
print(df.shape)
df.head(5)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
x_test
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
x_test
y_pred=model.predict(x_test)
model.predict_proba(x_test)
print('predicted response:',y_pred,sep='\n')
#y_pred=y_pred.reshape(-1,1)
r2=model.score(x_test,y_test)
b0=model.intercept_
b1=model.coef_
print('coefficient of Determination:', r2)
print('==========================================')
print('intercept: =b0', b0)
print('==========================================')
print('slope: =b0', b1)
df=pd.DataFrame({'Actual': y_test,'Predicted': y_test})
df
import math
def sigmoid(x):
    return 1 / (1+math.exp(-x))
def prediction_function(age):
    z = 0.182 * age - 5.98
    y = sigmoid(z)
    return y
age = 24
prediction_function(age)
age = 32
prediction_function(age)
import pickle
with open('model_pickle','wb')as f:
    pickle.dump(model,f)
with open('model_pickle','rb')as f:
    mp=pickle.load(f)
mp.predict(x_test)

