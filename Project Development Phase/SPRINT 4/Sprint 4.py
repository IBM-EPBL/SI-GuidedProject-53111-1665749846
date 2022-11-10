#TRAINING THE LINEAR REGRESSION MODEL AND PLOTTING THE FORECAST GRAPH

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