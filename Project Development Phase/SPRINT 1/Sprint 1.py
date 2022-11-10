#DATA PREPROCESSING

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

