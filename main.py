import time
import datetime as datetime
import pandas as pd
import numpy as np
import re
import requests
import mplfinance as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns

print('-------------------------Start: Answer 1.a ----------------------------------------------------------------------------')
#Answer 1 Real-world scenario: The project should use a real-world dataset and include a reference of their source in the report (10)
# Working with S&P 500 companies historical prices and fundamental data. https://www.kaggle.com/dgawlik/nyse
# Dataset used are
#     Alphavantage for fetching real time Stock data: https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&outputsize=full&apikey=9AZIN6Q78VVQXW5H
#     prices.csv : This has historical prices for over 500 companies ranging from 4th Jan 2010 - 31st Dec 2016
#     Securities.csv : This has details like Company name, Headquarter address, Inception Date and their Sector and Industry Classification


# Function to fetch latest price and plot graphs for each stock list passed
def Latest_StockPrices(Stock_List):
   for i in Stock_List:
       print(i)
       urlPart1 = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='
       urlPart2 = '{}&outputsize=full&apikey=9AZIN6Q78VVQXW5H'.format(i)
       url = urlPart1 + urlPart2
       print(url)
       data = requests.get(url)
       print(data)
       # Converting the data into a Dictionary which has two keys 'Meta' and 'Time Series (Daily)'
       dict1 = data.json()
       print(type(dict1))
       #print(dict1.keys())
       dict2 = dict1['Time Series (Daily)']
       #print(type(dict2))
       df_API = pd.DataFrame.from_dict(dict2)
       # Transpose the data frame to have dates as rows and other fields as columns
       df_API_T = df_API.transpose()
       df_DailyData = df_API_T.reset_index()
       df_DailyData.rename(columns={'index': 'Date',
                                    '1. open': 'open', '2. high': 'high',
                                    '3. low': 'low', '4. close': 'close',
                                    '5. volume': 'volume'}, inplace=True)

       # Removing the trailing H:M:S from a datetime object and converting it into string
       df_DailyData['Date'] = pd.to_datetime(df_DailyData['Date'])

       # All the numeric data is converted from Object to Float
       df_DailyData['open'] = df_DailyData['open'].astype(float)
       df_DailyData['high'] = df_DailyData['high'].astype(float)
       df_DailyData['low'] = df_DailyData['low'].astype(float)
       df_DailyData['close'] = df_DailyData['close'].astype(float)
       df_DailyData['volume'] = df_DailyData['volume'].astype(float)

       # Plotting a candlestick chart with this above data
       df_DailyData = df_DailyData.set_index('Date')  # Setting the Date as Index

       # Plot a Candlestick chart with Daily moving averages and volumns
       mpl.plot(df_DailyData['2021-06-01':], type='candle',
                title='{} Candlestick Chart:Latest Day "Price","Volume"&"Moving Average"'.format(i),
                mav=(10), volume=True,style='yahoo')

# Function to convert an Object to Datetime and extracting Year of each date
def DateOperation(Date):
    Date1 = pd.to_datetime(Date)
    Year1 = pd.DatetimeIndex(Date1).year
    return(Date1,Year1)

print('-------------------------Start: Answer 2.a, 3.c, 4.a and 4.c.---------------------------------------------------------------')
# Answer 2.a Importing Data : Your project should make use of one or more of the following: Relational database, API or web scraping (10)
# Answer 3.c Analysing data : Make use of iterators (10)
# Answer 4.a Python : Define a custom function to create reusable code (10)
# Answer 4.c Python : Dictionary or Lists (10)
# Answer 6 Part I : Visualisation (Ignore all the Future warnings)

# For my solution I am using a Parameterised Function to call API's of 5 Tech Companies to fetching "Daily" Real time data from
# Alphavantage with my personal Key and present "Candle stick" chart with "Daily Moving Average" and "Volume" for all the 5 by using a For Loop

Stock_List = ['AAPL','GOOGL','MSFT','TWTR','AMZN']

#Latest_StockPrices(Stock_List) # Calling a parameterized Function and passing a List

print('-------------------------End: Answer 2.a, 3.c, 4.a and 4.c.-------------------------------------------------------------------')

print('------------------------------------Start:Answer 2.b -------------------------------------------------------------------------')

# Answer 2.b Importing Data - Import a CSV file into a Pandas DataFrame (10)
# Import S&P500 historical data and validating details for the same
df_Prices = pd.read_csv('prices.csv')
df_Securities = pd.read_csv('securities.csv')

print("--------------------Prices--------------------------")
# Understand the data set for Prices
print(df_Prices.shape)
print(df_Prices.head())
df_Prices.info()

print("----------------Securities---------------------------")
# Understand the data for Securities detail
print(df_Securities.shape)
print(df_Securities.head())
df_Securities.info()

print('----------------------------------End:Answer 2.b ---------------------------------------------------------------------------')

print('-----------------------------------Start:Answer 3.d -------------------------------------------------------------------------')
# Answer 3.d Analysing data : Merge DataFrames (10)
# Merging the Prices and Securities from above data to have complete data of Symbols, their prices and basic details

# Re-naming the 'Ticker' in Securities data to 'symbol' which acts as Primary key to perform "Full Join" on Securities and Prices dataset.
df_Securities.rename(columns={'Ticker symbol': 'symbol',
                              'Date first added': 'Inception Date',
                              'Security': 'Company Name'},
                     inplace=True)

df_merged = pd.merge(df_Securities, df_Prices, on='symbol', how='outer')
df_merged.info()
print(df_merged.head())
df_merged = df_merged[['CIK', 'symbol', 'Company Name', 'Address of Headquarters', 'GICS Sector', 'GICS Sub Industry',
                        'Inception Date', 'SEC filings', 'date', 'open', 'close', 'low', 'high',
                        'volume']]
# Understanding the nuances of Merged data
df_merged.info()
print(df_merged.head())
print(df_merged.notnull().count())

print('-----------------------------------------End:Answer 3.d -------------------------------------------------------------------------')


print('----------------------------------Start:Answer 3.b,4.b & 6 Part II -------------------------------------------------------------------')
# Answer 3.b Analysing data - Replace missing values or drop duplicates (10)
# Answer 4.b Python - Numpy(10)
# Answer 6 Part II : Visualize the count of companies based on Inception Date

# Answer 3.b,4.b :Analysis on Securities data reveals that Inception date has the least count (377) hence using iterations to fill them as 'Not Defined'
df_Securities.info()
Visual1 = []
Visual1 = list(df_Securities['Inception Date']) # To be used for visualisation

# Using Numpy function Where to validate the value in Numpy and if it is Null then replace the same with "Not Defined"
df_Securities['Inception Date'] = np.where(df_Securities['Inception Date'].isnull(), 'Not Defined', df_Securities['Inception Date'])

# Count of Inception Date now shows 505 as other fields
df_Securities.info()
print(df_Securities['Inception Date'].value_counts().sort_index())
print(df_Securities['Inception Date'].dtypes)


# Answer 6.a Part II: Visualize the count of companies based on Inception Date
Inception_Date, year = DateOperation(Visual1)
Visual_Year = year.value_counts()
x = list(Visual_Year.index)
y = list(Visual_Year)

fig, ax = plt.subplots()
width = 0.75 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="orange")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
for i, v in enumerate(y):
    ax.text(v + .25, i + .25, str(v), color='orange', fontweight='bold') #add value labels into bar
plt.title('Launch year vs Count of Companies')
plt.xlabel('Count of Companies')
plt.ylabel('Inception Year')
plt.show()

print('---------------------------------End:Answer 3.b,4.b & 6 Part II--------------------------------------------------------------')


print('--------------------------------Start:Answer 3.a, 3.c -------------------------------------------------------------------------')

# Answer 3.a Analysing data - Your project should use Regex to extract a pattern in data (10)
# Answer 3.c Make use of iterators (10)


# Using Securities data 'Address of Headquarters' and fetch the City for the same
Regex1 = r"\w+\s?\w*$"
City = []
for i in range(len(df_Securities['Address of Headquarters'])):
    S1 = str(df_Securities['Address of Headquarters'][i])
    City.append(re.findall(Regex1, S1))
df_Securities['City'] = City

print(df_Securities.head(50))
df_Securities.info()

# print('----------------------------------City Counts--------------------------------------------')
# df_City = pd.DataFrame(City)
# #print(df_City.types)
# print(pd.values_count(df_City.values.flatten()))

print('--------------------------------End:Answer 3.a, 3.c -------------------------------------------------------------------------')


print('----------------------------------Start: Answer 5  -------------------------------------------------------------------------')
# Work with Prices data to use ML - Regression Algo
# Filtering the Prices dataframe on a particular symbol for AMAZON = AMZN
selected_symbol = ['AMZN']
df_Prices_AMZN = df_Prices[df_Prices['symbol'].isin(selected_symbol)]
df_Prices_AMZN.info()

df_Prices_AMZN['date']= pd.to_datetime(df_Prices_AMZN['date'])

# As machine Algo works only on Numerical data then converting Data which is in string format to float and droping Symbol as the same is a redundent column

# ##try:
#     df_Prices_AMZN['date'] = pd.to_datetime(df_Prices_AMZN['date'], format='%d-%m-%Y')
# except Exception as e:
#     df_Prices_AMZN['date'] = pd.to_datetime(df_Prices_AMZN['date'], format='%Y-%m-%d')
# print(df_Prices_AMZN.dtypes)

# df_Prices_AMZN['date'] = int(df_Prices_AMZN['date'].strftime())
# df_Prices_AMZN['date'] = df_Prices_AMZN['date'].astype(float)
# df_Prices_AMZN['date'].apply(lambda x: float(x))


print(df_Prices_AMZN.dtypes)

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encoding Dates to be unique by passing Label_encoder
df_Prices_AMZN['date'] = label_encoder.fit_transform(df_Prices_AMZN['date'])
print(df_Prices_AMZN['date'].unique())
df_Prices_AMZN['date'].apply(lambda x: float(x))

df_Prices_AMZN1 = df_Prices_AMZN[['date','open','close','low','high','volume']]
print('---------Test New Name -------------------------')
print(df_Prices_AMZN1.head())

print('Printing the Co-Relation Matrix')
corrmat = df_Prices_AMZN1.corr()
print(corrmat)

# Answer 6.a Part II: Visualize the Correlation Heat map with Seaborn
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(df_Prices_AMZN1[top_corr_features].corr(), annot=True, cmap="RdYlGn")


#With Co-relation metrics its evident that Volumn is least corelated with any other feature.
# Hence setting up a ML algo to check for predictions of "Volumn" (Target) with other columns (Feature)
X = df_Prices_AMZN1.drop('volume', axis=1).values  # Feature
y = df_Prices_AMZN1['volume'].values  # Target

print(type(X))
print(type(y))
print(X.shape)
print(y.shape)
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Instantiate a Decision Tree classifier: tree
# tree = DecisionTreeClassifier()
# tree.fit(X_train, y_train)
# print("Decision Tree:",tree.score(X_test, y_test))
#
#
# # Instantiate a logistic regression classifier: logreg
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# y_Logreg = logreg.predict(X_test)
# print("Logistic Regression score is :",logreg.score(X_test, y_test))

# Using Lasso to identify the Most important predictor for a close price is 'Open' price
names = df_Prices_AMZN1.drop('volume', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()


# Using Lasso Regression for Regularize
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print('Lasso Regression score is: ',lasso.score(X_test, y_test))


# Setup the parameters and distributions to sample from: param_dist
# param_Lasso = {"alpha" : [0.1,0.4,0.7,1],
#               "fit_intercept": [True, False],
#               "normalize": [True, False],
#               "copy_X": [True, False],
#               # "n_jobs": [1],
#               # "positive": [True, False]
#               }
# find optimal alpha with grid search
#
# lassreg = GridSearchCV(lasso, param_Lasso,cv=None)
#
# # Fit it to the data
# lassreg.fit(X,y)
#
# # Print the tuned parameters and score
# print("Tuned Lasso Regression Parameters: {}".format(lassreg.best_params_))
# print("Best score is {}".format(lassreg.best_score_))
#----------------------------------------------------------------


# Hyperparameter tuning for Lasso Regression
alpha = [0.001, 0.01, 0.1, 1]
param_Lasso1 = dict(alpha=alpha)
grid_lasso = GridSearchCV(estimator=lasso, param_grid=param_Lasso1, scoring='r2', verbose=1, n_jobs=-1, cv=10)
grid_Lasso_result = grid_lasso.fit(X_train, y_train)

print('Best Lasso Params grid_Lasso_result: ', grid_Lasso_result.best_params_)
print('Best Lasso Score grid_Lasso_result: ', grid_Lasso_result.best_score_)



# Using Ridge Regression for Regularize
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge Regression Score is :", ridge.score(X_test, y_test))


# Hyperparameter tuning for Ridge Regression
alpha = [0.001, 0.01, 0.1, 1]
param_grid = dict(alpha=alpha)
grid = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='r2', verbose=1, n_jobs=-1, cv=10)
grid_result = grid.fit(X_train, y_train)
print('Best Params for Ridge Regression: ', grid_result.best_params_)
print('Best Score for Ridge Regression : ', grid_result.best_score_)


# Trying Linear ElasticNet
regr =ElasticNet()
regr.fit(X_train,y_train)
print('ElasticNet Regression score is :',regr.score(X_test, y_test))


#Trying Linear Regression score
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
print("Linear Regression score is :",reg_all.score(X_test, y_test))

# Hyper-parameter tuning for Linear Regression
# Setup the parameters and distributions to sample from: param_dist
param_dist = {"fit_intercept": [True, False],
              "normalize": [True, False],
              "copy_X": [True, False],
              "n_jobs": [1,3,5],
              "positive": [True, False]
              }
logreg_cv = GridSearchCV(reg_all, param_dist, cv=None)

# Fit it to the data
logreg_cv.fit(X_train,y_train)

# Print the tuned parameters and score
print("Best Parameters for Linear Regression: ",logreg_cv.best_params_)
print("Best Score for Linear Regression: ",logreg_cv.best_score_)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

print('-------------------------End: Answer 5  -------------------------------------------------------------------------')


#Visual_City = City.value_counts()
# A graph to show count of Companies in S&P500 and their cities
# df_SecByDate = df_Securities.groupby(['City'],as_index= False)
# print(df_SecByDate.dtypes)

#-----------------------------------------------------------------------------------------



# Find the count of Ticker Symbol in Securities data
#print(df_Securities['Ticker symbol'].value_counts())
#print(df_Securities['Ticker symbol'].unique())


# # Data Cleaning and Validation on Securities
# print(df_Securities['symbol'].isnull().count())
# print(df_Securities.columns)


# for i in df_Securities.columns[0:]:
#     print("Columns are" +df_Securities.columns[i])
#     print("Null values in "+df_Securities[i].count())
#     #print("Null values in " + df_Securities[i] "is ="+df_Securities[i].isnull.count() )







# df.describe() count, mean, std etc

#print(df_Prices.groupby('date').count(unique()))
#Function to get current price of any Symbol as per Yahoo Finance#Function to get current price of any Symbol as per Yahoo Finance
# query_string = 'https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1606141300&period2=1637677300&interval=1d&events=history&includeAdjustedClose=true'


# Positional formatting with {}  and f- strings to give the change in percentage and Symbol with maximum positive and negative change percentage

# tool="Unsupervised algorithms"
# goal="patterns"
# print("{title} try to find {aim} in the dataset".format(title=tool, aim=goal))

### Try and EXCEPT CODE : Try would test the give code and Except would bring out the error message when the same is not running


#As you've come to appreciate, there are many steps to building a model, from creating training and test sets, to fitting a classifier or regressor, to tuning its parameters, to evaluating its performance on new data. Imputation can be seen as the first step of this machine learning process, the entirety of which can be viewed within the context of a pipeline. Scikit-learn provides a pipeline constructor that allows you to piece together these steps into one process and thereby simplify your workflow.