import sys
import numpy
import csv
from sklearn.model_selection import train_test_split
import pandas as pd

traindic = {}
testdic = {}

def createTrainTestFolds(p1):
	for i in range(0,10):
			traindic[i], testdic[i] = train_test_split(p1, test_size=0.2)


POSdf = pd.read_csv('./POStrainclean.csv')
d2 = pd.read_csv('./POStestclean.csv')

POSdf = POSdf.append(d2, ignore_index = True)

createTrainTestFolds(POSdf)

for i in range(0,10):
	
	print('Writing train{0} file'.format(i+1))
	traindic[i].to_csv('./Train/Train{0}.csv'.format(i+1), index = False)
	print('Writing test{0} file'.format(i+1))
	testdic[i].to_csv('./Test/Test{0}.csv'.format(i+1), index = False)



