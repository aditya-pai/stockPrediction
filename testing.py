###AUTHOR: ADITYA D PAI
###------TESTING AND BENCHMARKING SCRIPT FOR MODELS-------


import numpy as np
import warnings
import pandas as pd
from sklearn import preprocessing

###import modules as per benchmarking requirements

#from sklearn import cross_validation
#from sklearn.neural_network import MLPClassifier
#from sklearn.cross_validation import KFold
#from sklearn.linear_model import LogisticRegression
#from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.neighbors.nearest_centroid import NearestCentroid
#import matplotlib.pyplot as plt
#from matplotlib import style
#import time
#start_time = time.time()


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)



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


def buildDataSet():

	data_df = pd.DataFrame.from_csv("keystats_reduced.csv")
	data_df=data_df.reindex(np.random.permutation(data_df.index))
	data_df=data_df.replace("NaN",0).replace("N/A",0)
	X=np.array(data_df[features].values)
	y= (data_df["Status"].replace("underperform",0).replace("outperform",1).values.tolist())
	X=preprocessing.scale(X)

	Z=np.array(data_df[["stock_p_change","sp500_p_change"]])
	return X,y,Z



def analysis():

	test_size = 1000
	amount=10000
	inv_made=0
	market_earn=0
	pred_invest=0

	X,y,Z=buildDataSet()
	y=np.array(y)
	print(len(X))

	###algortihm selection for testing and benchmarking

	#clf=svm.SVC(kernel="poly",degree=10,C=1)
	clf=RandomForestClassifier(max_features=None, oob_score=True)
	#clf=GradientBoostingClassifier()
	#kf_total = cross_validation.KFold(len(X), n_folds=2,  shuffle=True, random_state=4)
	#clf=NearestCentroid(metric='euclidean', shrink_threshold=None)
	#clf=LogisticRegression()
	#scores=[]
	#scores = cross_validation.cross_val_score(clf, X[:-test_size],y[:-test_size], cv=5)
	#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
	clf.fit(X[:-test_size],y[:-test_size])


	#for train_indices, test_indices in kf_total:
		#clf.fit(X[train_indices], y[train_indices])
	#scores.append(clf.fit(X[train_indices], y[train_indices]).score(X[test_indices],y[test_indices]))
	
	###accuracy computation

	correct = 0
	inv_made = 0
	market_earn =0
	pred_invest=0
	for x in range(1,test_size+1):
		if clf.predict(X[-x])[0] == y[-x]:
			correct+=1

		if clf.predict(X[-x])[0] == 1:
			invest_return=amount+(amount*(Z[-x][0]/100))
			market_return=amount+(amount*(Z[-x][1]/100))

			inv_made+=1
			market_earn+=market_return
			pred_invest+=invest_return
	print("Accuracy:",(correct/test_size)*100.00)
	print("Total Trades:",inv_made)
	print("Ending with Strategy",pred_invest)
	print("Ending with Market",market_earn)

	###performance computation by backtesting
	compared = ((pred_invest-market_earn)/market_earn)*100.

	market = inv_made*amount
	avg_market=((market_earn- market)/ market)*100
	avg_strat=((pred_invest- market)/ market)*100
	print("Compared to market we earn",str(compared)+"% more")
	print("Average investment return:",str(avg_strat)+"%")
	print("Average market return:",str(avg_market)+"%")
	#print(scores)
	#print("--- %s seconds ---" % (time.time() - start_time))

analysis()