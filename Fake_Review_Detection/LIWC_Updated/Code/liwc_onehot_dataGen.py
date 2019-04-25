
import sys
import numpy
import csv
import pandas as pd
from sklearn.model_selection import train_test_split

#reading the CSV file containing the LIWC results
p1 = pd.read_csv('../Data/liwc2015.csv')


traindic = {}
testdic = {}

class score():
	
	def __init__(self):
	
		self.mean={}
		self.std={}
		pass
	
	def fit(self,train_df):
	
		train_df = train_df.drop('Column1', axis = 1)
		train_df = train_df.drop('Column2', axis = 1)
		train_df = train_df.drop('Column3', axis = 1)
		train_df = train_df.drop('Column4', axis = 1)
		columns=list(train_df)
		for i in columns:
			self.mean[i]=train_df[i].mean()
			self.std[i]=train_df[i].std()
			for j,row in train_df.iterrows():
				if self.std[i] != 0:
					val=(row[i] - self.mean[i]) / self.std[i]
				else:
					val=(row[i] - self.mean[i])
				train_df.set_value(j,i,val)
				
		return train_df
		pass
	
	def test(self,test_df):
	
		test_df = test_df.drop('Column1', axis = 1)
		test_df = test_df.drop('Column2', axis = 1)
		test_df = test_df.drop('Column3', axis = 1)
		test_df = test_df.drop('Column4', axis = 1)
		col=list(test_df)
		for i in col:
			for k,row in test_df.iterrows():
				if self.std[i] != 0:
					va=(row[i]-self.mean[i])/self.std[i]
				
				else:
					va=(row[i]-self.mean[i])
				test_df.set_value(k,i,va)
		
		return test_df			
		pass

class enc():

	def __init__(self):
		pass
	
	def create_traintest(self,p1):
		for i in range(0,10):
			traindic[i], testdic[i] = train_test_split(p1, test_size=0.2)
		
	def onehot_zscore(self, temptraindf, temptestdf):
		s = score()
		
		zscoretrain = s.fit(temptraindf)
		traincol2 = pd.get_dummies(temptraindf['Column2'])
		traincol3 = pd.get_dummies(temptraindf['Column3'])
		traincol4 = pd.get_dummies(temptraindf['Column4'])
		zscoretrain = zscoretrain.join(traincol2)
		zscoretrain = zscoretrain.join(traincol3)
		zscoretrain = zscoretrain.join(traincol4)
		zscoretrain = zscoretrain.join(temptraindf['Column1'])
		
		zscoretest = s.test(temptestdf)
		testcol2 = pd.get_dummies(temptestdf['Column2'])
		testcol3 = pd.get_dummies(temptestdf['Column3'])
		testcol4 = pd.get_dummies(temptestdf['Column4'])
		zscoretest = zscoretest.join(testcol2)
		zscoretest = zscoretest.join(testcol3)
		zscoretest = zscoretest.join(testcol4)
		zscoretest = zscoretest.join(temptestdf['Column1'])
		
		return zscoretrain, zscoretest

d = {'truthful': 1, 'deceptive': 0}
p1['Column1'].replace(d,inplace = True)

p1 = p1.drop('Column5', axis = 1)

e = enc()
e.create_traintest(p1)

for i in range(0,10):
	traintemp, testtemp = e.onehot_zscore(traindic[i], testdic[i])
	print('Writing train{0} file'.format(i+1))
	traintemp.to_csv('../Result/Train/Train{0}.csv'.format(i+1), index = False)
	print('Writing test{0} file'.format(i+1))
	testtemp.to_csv('../Result/Test/Test{0}.csv'.format(i+1), index = False)
	





