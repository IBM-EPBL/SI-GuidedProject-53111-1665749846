#PREPARING THE SUPERVISED DATA TO GIVE IT TO THE MACHINE LEARNING MODEL

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

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

