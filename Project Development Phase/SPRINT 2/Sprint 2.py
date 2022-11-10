#VISUALIZATION OF MONTHLY SALES AND MONTHLY SALES DIFFERENCE BY YEAR

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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





