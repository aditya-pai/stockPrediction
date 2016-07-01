###AUTHOR: ADITYA D PAI
###------SCRIPT FOR GEREATING IDEAL INVESTMENTS-------


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
import pandas as pd
from matplotlib import style
import statistics
from collections import Counter
import warnings
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

  #initialize list of features

features = ['DE Ratio', 'Trailing P/E', 'Price/Sales', 
      'Price/Book', 'Profit Margin', 'Operating Margin', 
      'Return on Assets', 'Return on Equity', 'Revenue Per Share', 
      'Market Cap', 'Enterprise Value', 'Forward P/E',
       'PEG Ratio', 'Enterprise Value/Revenue', 'Enterprise Value/EBITDA', 
       'Revenue', 'Gross Profit', 'EBITDA', 
       'Net Income Avl to Common ', 'Diluted EPS', 'Earnings Growth', 
       'Revenue Growth', 'Total Cash', 'Total Cash Per Share', 
       'Total Debt', 'Current Ratio', 'Book Value Per Share', 
       'Cash Flow', 'Beta', 'Held by Insiders', 
       'Held by Institutions', 'Shares Short (as of', 'Short Ratio', 
       'Short % of Float', 'Shares Short (prior ']


def performanceMargin(stock, sp500):       ###identify outperforming stocks
  #set performance margin
  margin = 15
  difference = stock - sp500
  if difference > margin:
    return 1
  else:
    return 0

def buildDataSet():

  ###select dataset     
  # data_df = pd.DataFrame.from_csv("key_stats_reduced_enhanced.csv")
  data_df = pd.DataFrame.from_csv("key_stats_full_enhanced.csv")
  # shuffle data:
  data_df = data_df.reindex(np.random.permutation(data_df.index))
  data_df = data_df.replace("NaN",0).replace("N/A",0)


  #get outperforming stocks
  data_df["Status2"] = list(map(performanceMargin, data_df["stock_p_change"], data_df["sp500_p_change"]))
  
  X = np.array(data_df[features].values)#.tolist())
  X = preprocessing.scale(X)
  
  y = ( data_df["Status2"].values.tolist())

  return X,y



def analysis():       ###function for generating outperforming stocks

  #initial parameters
  test_size = 1
  amount = 10000 
  inv_made = 0
  market_earn = 0
  pred_invest = 0
  X,y= buildDataSet()

  #Create Model
  clf=RandomForestClassifier(max_features=None, oob_score=True)
  clf.fit(X[:-test_size],y[:-test_size]) 

  data_df = pd.DataFrame.from_csv("forward_sample_full.csv")
  data_df = data_df.replace("NaN",0).replace("N/A",0)
  X = np.array(data_df[features].values)
  X = preprocessing.scale(X)
  Z = data_df["Ticker"].values.tolist()
  invest_list = []

  #return list of outperforming stocks
  for i in range(len(X)):
    p = clf.predict(X[i])[0]
    if p == 1:
      invest_list.append(Z[i])

  return invest_list

def getInvestments():      ###function for filtering ideal investments

  final_list = []
  loops = 3

  #generate consolidated list of outperforming stocks
  for x in range(loops):
    print("Iteration: "+str(x+1))
    stock_list = analysis()
    for e in stock_list:
      final_list.append(e)
  x = Counter(final_list)
  print(x)
  print('_'*120)
  invest_in=[]
  count=0

  #generate filtered list of stocks for investment
  for each in x:
    if (x[each] >2) and count<15:
      invest_in.append(each)
      count+=1
  print ("Strategized Investments: "+str(sorted(invest_in)))

getInvestments()



