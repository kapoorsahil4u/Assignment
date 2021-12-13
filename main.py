import time
import datetime
import pandas as pd
import numpy as np
import re
import requests
import mplfinance as mpl
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
#from mplfinance import candlestick_ohlc
import matplotlib.dates as mpdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

print('-------------------------Start: Answer 1.a ----------------------------------------------------------------------------')
#Answer 1 Real-world scenario: The project should use a real-world dataset and include a reference of their source in the report (10)
# Working with S&P 500 companies historical prices and fundamental data. https://www.kaggle.com/dgawlik/nyse
# Dataset used are
#     prices.csv : This has historical prices for over 500 companies ranging from 4th Jan 2010 - 31st Dec 2016
#     Securities.csv : This has details like Company name, Headquarter address, Inception Date and their Sector and Industry Classification


# Answer 2.a Importing Data : Your project should make use of one or more of the following: Relational
# database, API or web scraping (10)

# For my solution I am using a API method and fetching historical data for AMZN from Alphavantage with Key
print('-------------------------Start: Answer 2.a, Answer 4.c.---------------------------------------------------------------')

data= requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AMZN&outputsize=full&apikey=9AZIN6Q78VVQXW5H')
print(data)
# Converting the data into a Dictionary which has two keys 'Meta' and 'Time Series (Daily)'
dict1 = data.json()
print(type(dict1))

# Answer 4.c Python : Dictionary or Lists (10)
# As the above is a nested Dict with "Meta" and "Time Series (Daily)" keys The following is extracting 'Time Series (Daily)'
# key from the above to a new Dictionary

dict2 = dict1['Time Series (Daily)']
print(type(dict2))

# Having operation on Dictionary 2 where converting dict to a data frame
df_API = pd.DataFrame.from_dict(dict2)
df_API.info()
print(df_API.head())
print(df_API.shape)

# Transpose the data frame to have dates as rows and other fields as columns
df_API_T = df_API.transpose()
df_API_T.info()

df_DailyData = df_API_T.reset_index()
df_DailyData.rename(columns={'index': 'Date',
                       '1. open': 'open','2. high': 'high',
                       '3. low': 'low','4. close': 'close',
                       '5. volume': 'volume'}, inplace=True)

# Removing the trailing H:M:S from a datetime object and converting it into string
df_DailyData['Date'] = pd.to_datetime(df_DailyData['Date'])

# All the numeric data is converted from Object to Float
df_DailyData['open']= df_DailyData['open'].astype(float)
df_DailyData['high']= df_DailyData['high'].astype(float)
df_DailyData['low']= df_DailyData['low'].astype(float)
df_DailyData['close']=df_DailyData['close'].astype(float)
df_DailyData['volume']=df_DailyData['volume'].astype(float)
print(df_DailyData['Date'][10])
print(df_DailyData.head())
df_DailyData.info()
print(df_DailyData.shape)

# Plotting a candlestick chart with this above data
df_DailyData = df_DailyData.set_index('Date') #Setting the Date as Index

# Plot a Candlestick chart with Daily moving averages and volumns
mpl.plot(df_DailyData['2021-06-01':], type='candle', title = 'AMAZON Daily Chart',mav=(20), volume= True, style= 'yahoo')

print('-------------------------End: Answer 2.a, Answer 4.c.---------------------------------------------------------------')

print('-------------------------Start:Answer 2.b ---------------------------------------------------------------------------')

# Answer 2.b Importing Data - Import a CSV file into a Pandas DataFrame (10)
# Import S&P500 historical data and details

# Input_path  =
# Output_path =
# df_Prices = pd.read_csv(r'C:\Users\BYO\Desktop\UCD Course\Project Work\New York Stock Exchange_S&P 500 companies historical prices with fundamental data\prices.csv')
df_Prices = pd.read_csv('prices.csv')
# df_Securities = pd.read_csv(r'C:\Users\BYO\Desktop\UCD Course\Project Work\New York Stock Exchange_S&P 500 companies historical prices with fundamental data\securities.csv')
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

print('-------------------------End:Answer 2.b ---------------------------------------------------------------------------')

print('-------------------------Start:Answer 3.d -------------------------------------------------------------------------')

# Renaming the Ticker symbol to symbol so that Merge action to perform
df_Securities.rename(columns={'Ticker symbol': 'symbol',
                              'Date first added': 'Inception Date',
                              'Security': 'Company Name'},
                     inplace=True)

# Answer 3 d Analysing data -Merging datasets (10)
# Full Join (Outer Join)
df_merged = pd.merge(df_Securities, df_Prices, on='symbol', how='outer' )
df_merged.info()
print(df_merged.head())
df_merged = df_merged[['CIK', 'symbol', 'Company Name', 'Address of Headquarters', 'GICS Sector', 'GICS Sub Industry',
                        'Inception Date', 'SEC filings', 'date', 'open', 'close', 'low', 'high',
                        'volume']]
df_merged.info()
print(df_merged.head())
print(df_merged.notnull().count())

print('-------------------------End:Answer 3.d -------------------------------------------------------------------------')


print('-------------------------Start:Answer 3.b, Answer 4.b --------------------------------------------------------------')

# Answer 3.b Analysing data - Replace missing values or drop duplicates (10)
# Answer 4.b Python - Numpy(10)
# Inception date has the least count of all the columns (377) hence using iterations to fill them as 'Not Defined'
df_Securities['Inception Date'] = np.where(df_Securities['Inception Date'].isnull(), 'Not Defined', df_Securities['Inception Date'])

# Count of Inception Date now shows 505 as other fields
df_Securities.info()

print('-------------------------End:Answer 3.b, Answer 4.b --------------------------------------------------------------')


print('-------------------------Start:Answer 3.a, 3.c -------------------------------------------------------------------------')

# Answer 3.a Analysing data - Your project should use Regex to extract a pattern in data (10)
# Answer 3.c Make use of iterators (10)
# Trying to use Securities data 'Address of Headquarters' and fetch the City for the same

Regex1 = r"\w+\s?\w*$"
City = []
for i in range(len(df_Securities['Address of Headquarters'])):
    S1 = str(df_Securities['Address of Headquarters'][i])
    City.append(re.findall(Regex1, S1))
df_Securities['City'] = City

print(df_Securities.head(50))
df_Securities.info()

print('-------------------------End:Answer 3.a, 3.c -------------------------------------------------------------------------')

print('-------------------------Start: Answer 5  -------------------------------------------------------------------------')
# Work with Prices data to use ML - Regression Algo
# Filtering the Prices dataframe on a particular symbol for AMAZON = AMZN
selected_symbol = ['AMZN']
df_Prices_AMZN = df_Prices[df_Prices['symbol'].isin(selected_symbol)]
df_Prices_AMZN.info()
print(df_Prices_AMZN.dtypes)

# As machine Algo works only on Numerical data then converting Data which is in string format to float and droping Symbol as the same is a redundent column

# ##try:
#     df_Prices_AMZN['date'] = pd.to_datetime(df_Prices_AMZN['date'], format='%d-%m-%Y')
# except Exception as e:
#     df_Prices_AMZN['date'] = pd.to_datetime(df_Prices_AMZN['date'], format='%Y-%m-%d')
# print(df_Prices_AMZN.dtypes)

# df_Prices_AMZN['date'] = int(df_Prices_AMZN['date'].strftime())
# df_Prices_AMZN['date'] = df_Prices_AMZN['date'].astype(float)
# df_Prices_AMZN['date'].apply(lambda x: float(x))

#(df.A).apply(lambda x: float(x))
print(df_Prices_AMZN.dtypes)

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

# Encode labels in column 'species'.
df_Prices_AMZN['date'] = label_encoder.fit_transform(df_Prices_AMZN['date'])
print(df_Prices_AMZN['date'].unique())
df_Prices_AMZN['date'].apply(lambda x: float(x))
print(df_Prices_AMZN.dtypes)

# Dropping Close as this is the target value and Symbol as this is a string
X = df_Prices_AMZN.drop(columns=['close', 'symbol'])
y = df_Prices_AMZN['close'].values

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
# Trying to fit KNN model
# knn = KNeighborsClassifier(n_neighbors=8)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(knn.score(X_test, y_test))

# Trying to fit LinearRegression model
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
print(reg_all.score(X_test, y_test))

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

print('-------------------------End: Answer 5  -------------------------------------------------------------------------')

print('-------------------------Start: Answer 6 and Answer 7  -------------------------------------------------------------------------')
# A graph to show count of Companies in S&P500 and their cities
df_SecByDate = df_Securities.groupby(['City'],as_index= False)
print(df_SecByDate.dtypes)


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