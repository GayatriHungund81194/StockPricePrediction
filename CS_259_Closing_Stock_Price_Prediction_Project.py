#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
import matplotlib.pyplot as plt
import pandas_datareader as preader
import quandl
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics
from sklearn import preprocessing


# In[2]:


#sampling 
quandl.ApiConfig.api_key = "8LScULjVwsmvzyZKUzxG"
df = quandl.get("EOD/DIS", start_date="1986-10-01", end_date="2019-05-16")
df.describe()


# In[3]:


#A subset of data is displayed to see the various values in a dataset
#After selecting data it can be noticed that the entries for "2019-01-05" and "2019-01-06" are missing
#The reason behind the missing values is that the New York Stock exchange is closed on weekend and public holidays
df.loc['2019'].head()
#df.keys


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
#The frequency distribution will help to study how the data is spread 
#The frequency distribution of Open price shows that the frequency of prices in approximate range 35 to 38
#are most popular whereas other factors such as dividend and split are taken into consideration 
#while calculating the adjusted open price of stock this range changes to approximately 22 to 28
#This variation can also be plotted for other dataset features
#While trading for next day or investing on the stock on next day, the variation in the adjusted and real prices
#of stock should be studied as the adjusted prices are more accurate as they consider various factors 
#after the stock market closes for the day.
plt.figure(figsize=(15, 5))
plt.title('Distribution of Open Price and Adjusted Open Price')
plt.xlabel('Open Price and Adjusted Open Price')
plt.ylabel('Count')
sns.distplot(df['Open'], hist=True)
plt.ylabel('Count')
sns.distplot(df['Adj_Open'], hist=True)
plt.legend(loc='best')

#Frequency distribution of high price and adjusted high price is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of High Price and Adjusted High Price')
plt.xlabel('High Price and Adjusted High Price')
plt.ylabel('Count')
sns.distplot(df['High'], hist=True)
plt.ylabel('Count')
sns.distplot(df['Adj_High'], hist=True)

#Frequency distribution of low price and adjusted low price is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of Low Price and Adjusted Low Price')
plt.xlabel('Low Price and Adjusted Low Price')
plt.ylabel('Count')
sns.distplot(df['Low'], hist=True)
plt.ylabel('Count')
sns.distplot(df['Adj_Low'], hist=True)

#Frequency distribution of volume and adjusted volume is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of Volume and Adjusted Volume')
plt.xlabel('Volume and Adjusted Volume')
plt.ylabel('Count')
sns.distplot(df['Volume'], hist=True)
plt.ylabel('Count')
sns.distplot(df['Adj_Volume'], hist=True)

#Frequency distribution of split is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of Split')
plt.xlabel('Split')
plt.ylabel('Count')
sns.distplot(df['Split'], hist=True)


#Frequency distribution of dividend is plotted as follows
plt.figure(figsize=(15, 5))
plt.title('Distribution of Dividend')
plt.xlabel('Dividend')
plt.ylabel('Count')
sns.distplot(df['Dividend'], hist=True)


# In[5]:


#Removing the target variable for prediction
y = df.pop("Close")
print(y)
adj_y = df.pop("Adj_Close")


# In[6]:


#the relationship between the dataframe features can be determined by studying the bivariate 
#frequency distribution created by using pairplot
#Consider the relation between the open and high the scatter plot is almost linear which establishes a linear pattern.
sns.pairplot(df)

#check if the data frames contain any values that are null
print(df.isna().sum())
df = df.dropna(inplace=False)

#the values for these features are constant and have no change hence we drop the columns as the
#features do not count towrds making the predictions more accurate
# df = df.drop(columns=['Dividend'])
# df = df.drop(columns=['Split'])
df = df.dropna(inplace=False)


# In[7]:


#To check the co-relation between the stock market dataset features we have used correlation matrix
#As per the correlation matrix, we can decide if the value of selected features decrease or increase on change in 
#some other feature. 
#For example, the value of correlation coefficient is -0.3 for Volume and Open which means that 
#as the open price of stock increases the number of investors will decrease
#The matrix also demonstrates high correlation between the Open and High price of stock
#Also the split has negative or almost zero correlation with the other features of stock price data
#which means that the variable has no relation or extremely weak relation with other features
#Further considering the dividend also has weak relation with other features
df1 = df[['Open','High','Low','Volume','Dividend','Split']]
cor = df1.corr()
cor

#plt.set_xticklabels(['Open','High','Low','Volume','Dividend','Split'])


# In[13]:


#remove the target value from the dataset
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.33, random_state=1)
print("Training Set:")
print(X_train)
print("Closing Price of Stock in Training Set")
print(y_train)


# In[14]:


#checking the statistical distribution of data before normalizing the data 
#will help to know the minimum and maximum in the data
train_stats = X_train.describe()
print(train_stats)
# train_stats.pop("Close")
df3 = pd.DataFrame()
df3 = X_test
dates = pd.to_datetime(df3.index)

min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = min_max_scaler.fit(X_train)
df_train_norm = min_max_scaler.transform(X_train)


min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler = min_max_scaler.fit(X_test)
df_test_norm = min_max_scaler.transform(X_test)


# In[12]:


#normalize using z-score
# def normalizeData(data):
#     data=abs((data-train_stats['mean'])/train_stats['std'])
#     return data

# df_train_norm=normalizeData(X_train)
# df_test_norm=normalizeData(X_test)
# df_train_y=normalizeData(y_train)
# df_test_y=normalizeData(y_test)



#When the max depth is set to values in range 1 to 10, the MSE is high and about 3.5
#When the max depth is set to values greater than 50, the MSE is lowered and about 0.24
#The n_estimators set to 59 gives the best MSE value of 0.246 and the values grater than 59 increase the MSE to 0.2470
#As a multicore machine is used to execute the code the value of n_jobs is set to -1 to use all cores
warnings.filterwarnings("ignore")
print("\n\nRandomForestRegressor")
random_forest = RandomForestRegressor(random_state=1,n_estimators=59,max_depth=50,bootstrap=True,n_jobs=-1)
random_forest.fit(X_train,y_train)
y_randomForestPred = random_forest.predict(X_test)
randomForestMSE = mean_squared_error(y_test,y_randomForestPred)
print("MSE      = " + str(randomForestMSE))
randomForestR2 = r2_score(y_test,y_randomForestPred)
print("r2_score = " + str(randomForestR2))
#print("Explained Variance Score = " + str(explained_variance_score(y_test,y_randomForestPred)))
plt.figure(figsize=(20,20))
plt.scatter(dates,y_randomForestPred,label="Predicted Closing Stock Price")
plt.scatter(dates,y_test,c="#ADD8E6",label="Actual Closing Stock Price")
plt.title("Predicted Closing Price of Stock and Actual Closing Price of Stock using Random Forest Regressor",fontsize=20)
plt.xlabel('Time in Years',fontsize=20)
plt.ylabel('Stock Closing Price Predictions(Random Forest Regressor)',fontsize=20)
plt.legend(loc='best')
plt.show()

#The value of learning_rate is first set to 0.01 then the MSE is set to 0.2500
#The value of learning_rate is first set to 0.001 then the MSE is set to 0.2484
#The best MSE result obtained using Ada boost is 0.24949
print("AdaBoost Regressor")
regr_ada = AdaBoostRegressor(random_forest,n_estimators=30,random_state=1,learning_rate=0.000165)
regr_ada.fit(X_train,y_train)
regr_ada_pred = regr_ada.predict(X_test)
acc_ada_mse = mean_squared_error(y_test,regr_ada_pred)
print("MSE      = "+ str(acc_ada_mse))
r2_ada = r2_score(y_test,regr_ada_pred)
print("r2_score = " + str(r2_ada))
plt.figure(figsize=(20,20))
plt.scatter(dates,regr_ada_pred,label="Predicted Closing Stock Price")
plt.scatter(dates,y_test,c="orange",label="Actual Closing Stock Price")
plt.title("Predicted Closing Price of Stock and Actual Closing Price of Stock using ADA Boost Regressor",fontsize=20)
plt.xlabel('Time in Years',fontsize=20)
plt.ylabel('Stock Closing Price Predictions(AdaBoost-RandomForest Regressor)',fontsize=20)
plt.legend(loc='best')
plt.show()


print("Linear Regression")
lin = linear_model.LinearRegression(n_jobs=-1)
fit_data = lin.fit(X_train,y_train)
y_pred=fit_data.predict(X_test)
acc1 = mean_squared_error(y_test,y_pred)
print(acc1)
acc2 = r2_score(y_test,y_pred)
print(acc2)
plt.figure(figsize=(20,20))
plt.scatter(dates,y_pred,label="Predicted Closing Stock Price")
plt.scatter(dates,y_test,c="#DDA0DD",label="Actual Closing Stock Price")
plt.title("Predicted Closing Price of Stock and Actual Closing Price of Stock using Linear Regression",fontsize=20)
plt.xlabel('Time in Years',fontsize=20)
plt.ylabel('Stock Closing Price Predictions(Linear Regression)',fontsize=20)
plt.legend(loc='best')
plt.show()

#The MSE obtained by setting random splitter is 0.456 while the best split gives MSE of 0.431
#Setting max_depth to 10 gives the MSE as 0.390 on the other hand, setting the default value gives MSE 0.431
# The MSE is lowered to 0.432 and could not be lowered down further
print("Decision Tree Regressor")
from sklearn.tree import DecisionTreeRegressor
dtree_regressor = DecisionTreeRegressor(criterion='mse', splitter='best',random_state=1,presort=True)
dtree_regressor.fit(X_train,y_train)
dtree_pred=dtree_regressor.predict(X_test)
acc5 = mean_squared_error(y_test,dtree_pred)
print(acc5)
acc6 = r2_score(y_test,dtree_pred)
print(acc6)
plt.figure(figsize=(20,20))
plt.scatter(dates,y_pred,label="Predicted Closing Stock Price")
plt.scatter(dates,y_test,c="#B0E0E6",label="Actual Closing Stock Price")
plt.title("Predicted Closing Price of Stock and Actual Closing Price of Stock using Decision Tree Regressor",fontsize=20)
plt.xlabel('Time in Years',fontsize=20)
plt.ylabel('Stock Closing Price Predictions(Decision Tree Regressor)',fontsize=20)
plt.legend(loc='best')
plt.show()


# In[ ]:




