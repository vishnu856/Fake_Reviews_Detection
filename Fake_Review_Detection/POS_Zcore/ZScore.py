import pandas as pd
import numpy as np

class score():
	
	def __init__(self):
	
		self.mean={}
		self.std={}
		pass
	
	def fit(self,train_df):
	
		columns=list(train_df)
		for i in columns:
			self.mean[i]=train_df[i].mean()
			self.std[i]=train_df[i].std()
			for j,row in train_df.iterrows():
				if self.std[i] != 0:
					val=float(row[i] - self.mean[i]) / self.std[i]
				else:
					val=float(row[i] - self.mean[i])
				train_df.set_value(j,i,val)
				
		return train_df
		pass
	
	def test(self,test_df):
	
		col=list(test_df)
		for i in col:
			for k,row in test_df.iterrows():
				if self.std[i] != 0:
					va=float(row[i]-self.mean[i])/self.std[i]
				
				else:
					va=float(row[i]-self.mean[i])
				test_df.set_value(k,i,va)
		
		return test_df			
		pass

train_df = pd.read_csv("./Train1.csv")
test_df = pd.read_csv("./Test1.csv")

train_df = train_df.astype(np.float32)
test_df = test_df.astype(np.float32)

s = score()
train_df = s.fit(train_df)
test_df = s.test(test_df)


print(train_df)

train_df.to_csv("./Train_Result.csv", index = False)
test_df.to_csv("./Test_Result.csv", index = False)
