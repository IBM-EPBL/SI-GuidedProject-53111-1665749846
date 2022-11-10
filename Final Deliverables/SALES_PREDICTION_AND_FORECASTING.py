# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qUIv01SVYk5DfCoRcJiAZnbMKH-UXvqG
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense ,LSTM
from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint

#Reading Dataset File 

store_sales=pd.read_csv("train.csv")


#Dropping Unneccessary Coloumn

store_sales=store_sales.drop(['store','item'],axis=1)


#converting date from object datatype to dateTime datatype

store_sales['date']=pd.to_datetime(store_sales['date'])

#Converting date to a month period and sum the number of items of each month

store_sales['date']=store_sales['date'].dt.to_period('M')
monthly_sales=store_sales.groupby('date').sum().reset_index()

#Converting the resulting data coloumn into timeStamp datatype 

monthly_sales['date']=monthly_sales['date'].dt.to_timestamp()

#Visualisation

plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'],monthly_sales['sales'])
plt.xlabel('date')
plt.xlabel('sales')
plt.title('Monthly Customer Sales')
plt.show()

monthly_sales['sales_diff']=monthly_sales['sales'].diff()
monthly_sales=monthly_sales.dropna()




plt.figure(figsize=(15,5))
plt.plot(monthly_sales['date'],monthly_sales['sales'])
plt.xlabel('date')
plt.xlabel('sales')
plt.title('Monthly Customer Sales Difference')
plt.show()

supervised_data=monthly_sales.drop(['date','sales'],axis=1)

#preparing The Supervised Dataset

for i in range(1,13):
  col_name='month'+str(i)
  supervised_data[col_name]=supervised_data['sales_diff'].shift(i)
supervised_data=supervised_data.dropna().reset_index(drop=True)


#Testing and Training Data

train_data=supervised_data[:-12]
test_data=supervised_data[-12:]


scaler=MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)


x_train,y_train=train_data[:,1:],train_data[:,0:1]
x_test,y_test=test_data[:,1:],test_data[:,0:1]
y_train=y_train.ravel()
y_test=y_test.ravel()

#Make prediction dataframe to merge the predicted sales price of all trained algorithms

sales_dates=monthly_sales['date'][-12:].reset_index(drop=True)
predict_df=pd.DataFrame(sales_dates)


act_sales=monthly_sales['sales'][-13:].to_list()


#To Creat Linear Regression Model and Predict The Output

lr_model=LinearRegression()
lr_model.fit(x_train,y_train)
lr_pre=lr_model.predict(x_test)

lr_pre=lr_pre.reshape(-1,1)
lr_pre_test_set=np.concatenate([lr_pre,x_test],axis=1)
lr_pre_test_set=scaler.inverse_transform(lr_pre_test_set)


result_list=[]
for index in range(0,len(lr_pre_test_set)):
  result_list.append(lr_pre_test_set[index][0]+act_sales[index])
lr_pre_series=pd.Series(result_list,name='Linear Prediction')
predict_df=predict_df.merge(lr_pre_series,left_index=True,right_index=True)


lr_mse=np.sqrt(mean_squared_error(predict_df['Linear Prediction'],monthly_sales['sales'][-12:]))
lr_mae=mean_absolute_error(predict_df['Linear Prediction'],monthly_sales['sales'][-12:])
lr_r2=r2_score=(predict_df['Linear Prediction'],monthly_sales['sales'][-12:])

#Visualization of the prediction against the actual sales

plt.figure(figsize=(15,5))

#Actual Sales

plt.plot(monthly_sales['date'],monthly_sales['sales'])

#Predicted Sales

plt.plot(predict_df['date'],predict_df['Linear Prediction'])
plt.title('Customer sales Forecast Using LR Model')
plt.xlabel('date')
plt.ylabel('sales')
plt.legend(['Actual Sales','predicted Sales'])
plt.show()