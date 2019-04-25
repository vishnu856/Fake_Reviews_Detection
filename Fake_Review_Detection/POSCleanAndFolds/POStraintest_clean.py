import sys
import numpy
import csv

import pandas as pd

def POSclean(p1):
	p1['Target'] = ''
	p1.loc[p1['TRUTHFUL'] == 1, 'Target'] = 0
	p1.loc[p1['DECEPTIVE'] == 1, 'Target'] = 1
	p1 = p1.drop('TRUTHFUL', axis = 1)
	p1 = p1.drop('DECEPTIVE', axis = 1)

	return p1

trainp = pd.read_csv('./POS_Train.csv')
trainp['OTHER'] = 0
testp = pd.read_csv('./POS_Test.csv')



trainClean = POSclean(trainp)
testClean = POSclean(testp)

trainClean.to_csv('./POStrainclean.csv', index = False)
testClean.to_csv('./POStestclean.csv', index = False)
