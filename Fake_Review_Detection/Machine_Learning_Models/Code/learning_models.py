import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import time
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from itertools import combinations
from sklearn.metrics import accuracy_score
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class Learning_Models():


	def simple_majority(self,all_pred,ensemble_pair):

		no_data_points = len(all_pred[0])
		#print(no_data_points)

		comb = combinations([0, 1, 2, 3, 4], ensemble_pair) 


		list_comb = list(comb)

		all_pred_ens = []
		
		for k in range(0,len(list_comb)):
		
			pred = []
		
			#print(list_comb[k])
			for i in range(0,int(no_data_points)):
			
				vote = [0,0]
				for j in range(0,5):
					
					if j in list_comb[k]:

						if int(all_pred[j][i]) == 0:

							vote[0] = vote[0] + 1
						else:
							vote[1] = vote[1] + 1

				if vote[0] > vote[1]:
				
					pred.append(0)

				else:
					pred.append(1)

			all_pred_ens.append(pred)
			#print(pred)
		#print(all_pred_ens)
		#print(len(all_pred_ens))
		return all_pred_ens,list_comb

		
		
	def avg_scored_prob(self,all_prob,ensemble_pair):

		no_data_points = len(all_prob[0])
		
		comb = combinations([0, 1, 2, 3, 4], ensemble_pair) 

		list_comb = list(comb)

		all_pred_ens_prob = []
			
		for k in range(0,len(list_comb)):

			pred = []
			#print(list_comb[k])
			for i in range(0,int(no_data_points)):

				sum_class2 = 0

				sum_class4 = 0				

				for j in range(0,5):

					if j in list_comb[k]:
			
						sum_class2 += all_prob[j][i][0]

						sum_class4 += all_prob[j][i][1]
						#print(all_prob[j][i][0])
					

				if (sum_class2/ensemble_pair) > (sum_class4/ensemble_pair):

					pred.append(0)

				else:

					pred.append(1)
			
			
			all_pred_ens_prob.append(pred)

		return all_pred_ens_prob,list_comb
			
				
			






	def __init__(self):
		self.no_bagged_models = 5
		self.boosted_DT = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
		self.bagged_DT = BaggingClassifier(base_estimator=self.boosted_DT, n_estimators = self.no_bagged_models, max_samples=0.8, max_features=1, verbose=0)
		self.bagged_NB = BaggingClassifier(base_estimator=GaussianNB(), n_estimators = self.no_bagged_models, max_samples=0.8, max_features=1, verbose=0)
		self.bagged_SVM = BaggingClassifier(base_estimator=SVC(gamma="auto"), n_estimators = self.no_bagged_models, max_samples=0.8, max_features=1, verbose=0)
		self.bagged_NN = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators = self.no_bagged_models, max_samples=0.8, max_features=1, verbose=0)
		self.bagged_LR = BaggingClassifier(base_estimator=LogisticRegression(solver="lbfgs"), n_estimators = self.no_bagged_models, max_samples=0.8, max_features=1, verbose=0)
		
		
	def fit(self, train_df, target_data):
		
		self.bagged_DT.fit(train_df, target_data)
		self.bagged_NB.fit(train_df, target_data)
		self.bagged_SVM.fit(train_df, target_data)
		self.bagged_NN.fit(train_df, target_data)
		self.bagged_LR.fit(train_df, target_data)		
	
	def test(self, test_data_without_target, target_data):
		
		test_predictions_DT = self.bagged_DT.predict(test_data_without_target)
		
		#print(sklearn.metrics.classification_report(target_data, test_predictions_DT))
		
		test_predictions_NB = self.bagged_NB.predict(test_data_without_target)
		
		#print(sklearn.metrics.classification_report(target_data, test_predictions_NB))
		
		test_predictions_SVM = self.bagged_SVM.predict(test_data_without_target)
		
		#print(sklearn.metrics.classification_report(target_data, test_predictions_SVM))
		
		test_predictions_NN = self.bagged_NN.predict(test_data_without_target)
		
		#print(sklearn.metrics.classification_report(target_data, test_predictions_NN))
		
		test_predictions_LR = self.bagged_LR.predict(test_data_without_target)
		
		#print(sklearn.metrics.classification_report(target_data, test_predictions_LR))


	def ensemble(self,train_data_without_target, tar_data):

		sm = {}

		prob = {}

		train_predictions_DT = self.bagged_DT.predict(train_data_without_target)
		sm[0] = accuracy_score(tar_data, train_predictions_DT)
		prob[0] = accuracy_score(tar_data, train_predictions_DT)

		train_predictions_NB = self.bagged_NB.predict(train_data_without_target)
		sm[1] = accuracy_score(tar_data, train_predictions_NB)
		prob[1] = accuracy_score(tar_data, train_predictions_NB)

		train_predictions_SVM = self.bagged_SVM.predict(train_data_without_target)
		sm[2] = accuracy_score(tar_data, train_predictions_SVM)
		prob[2] = accuracy_score(tar_data, train_predictions_SVM)

		train_predictions_NN = self.bagged_NN.predict(train_data_without_target)
		sm[3] = accuracy_score(tar_data, train_predictions_NN)
		prob[3] = accuracy_score(tar_data, train_predictions_NN)

		train_predictions_LR = self.bagged_LR.predict(train_data_without_target)
		sm[4] = accuracy_score(tar_data, train_predictions_LR)
		prob[4] = accuracy_score(tar_data, train_predictions_LR)

		all_pred = []

		all_pred.append(list(train_predictions_DT))

		all_pred.append(list(train_predictions_NB))
		
		all_pred.append(list(train_predictions_SVM))

		all_pred.append(list(train_predictions_NN))

		all_pred.append(list(train_predictions_LR))
	
		ensemble_3_sm,pair_3 = self.simple_majority(all_pred,3)

		#print("\n3 Pair Simple Majority Voting\n")
		for i in range(0,len(ensemble_3_sm)):
			acc = accuracy_score(tar_data, ensemble_3_sm[i])
			#print(str(acc)+" "+str(pair_3[i]))

			sm[(pair_3[i])] = acc

		ensemble_5_sm,pair_5 = self.simple_majority(all_pred,5)

		#print("\n5 Pair Simple Majority Voting\n")
		for i in range(0,len(ensemble_5_sm)):
			acc = accuracy_score(tar_data, ensemble_5_sm[i])
			#print(str(acc)+" "+str(pair_5[i]))
			sm[(pair_5[i])] = acc

		ensemble_4_sm,pair_4 = self.simple_majority(all_pred,4)

		#print("\n4 Pair Simple Majority Voting\n")
		for i in range(0,len(ensemble_4_sm)):
			acc = accuracy_score(tar_data, ensemble_4_sm[i])
			#print(str(acc)+" "+str(pair_4[i]))
			sm[(pair_4[i])] = acc

		ensemble_2_sm,pair_2 = self.simple_majority(all_pred,2)

		#print("\n2 Pair Simple Majority Voting\n")
		for i in range(0,len(ensemble_2_sm)):
			acc = accuracy_score(tar_data, ensemble_2_sm[i])
			#print(str(acc)+" "+str(pair_2[i]))
			sm[(pair_2[i])] = acc

		scored_prob_DT = self.bagged_DT.predict_proba(train_data_without_target)

		scored_prob_NB = self.bagged_NB.predict_proba(train_data_without_target)

		scored_prob_SVM = self.bagged_SVM.predict_proba(train_data_without_target)

		scored_prob_NN = self.bagged_NN.predict_proba(train_data_without_target)

		scored_prob_LR = self.bagged_LR.predict_proba(train_data_without_target)

		all_prob = []

		all_prob.append(scored_prob_DT)

		all_prob.append(scored_prob_NB)

		all_prob.append(scored_prob_SVM)

		all_prob.append(scored_prob_NN)

		all_prob.append(scored_prob_LR)

		ensemble_5_prob,pairp_5 = self.avg_scored_prob(all_prob,5)

		#print("\n5 Pair Avg Scored Prob\n")
		for i in range(0,len(ensemble_5_prob)):
			acc = accuracy_score(tar_data, ensemble_5_prob[i])
			#print(str(acc)+" "+str(pairp_5[i]))
			prob[pairp_5[i]] = acc

		ensemble_4_prob,pairp_4 = self.avg_scored_prob(all_prob,4)

		#print("\n4 Pair Avg Scored Prob\n")
		for i in range(0,len(ensemble_4_prob)):
			acc = accuracy_score(tar_data, ensemble_4_prob[i])
			#print(str(acc)+" "+str(pairp_4[i]))
			prob[pairp_4[i]] = acc

		ensemble_3_prob,pairp_3 = self.avg_scored_prob(all_prob,3)
		#print("\n3 Pair Avg Scored Prob\n")
		for i in range(0,len(ensemble_3_prob)):
			acc = accuracy_score(tar_data, ensemble_3_prob[i])
			#print(str(acc)+" "+str(pairp_3[i]))
			prob[pairp_3[i]] = acc

		ensemble_2_prob,pairp_2 = self.avg_scored_prob(all_prob,2)
		#print("\n2 Pair Avg Scored Prob\n")
		for i in range(0,len(ensemble_2_prob)):
			acc = accuracy_score(tar_data, ensemble_2_prob[i])
			#print(str(acc)+" "+str(pairp_2[i]))
			prob[pairp_2[i]] = acc

		#print(sm)
		#print("\n\n")
		#print(prob)

		return sm,prob

	
		

		
class Solution():

	def fake_reviews_detection(self, train_data_path, test_data_path):		

		total_train_df = pd.read_csv(train_data_path)
		test_df = pd.read_csv(test_data_path)

		train_data_without_target = total_train_df.iloc[:, :-1]
		target_data = total_train_df.iloc[:,-1]

		train_df, crossval_train, target_data_train, crossval_target = sklearn.model_selection.train_test_split(train_data_without_target, target_data, test_size = 0.2)

		#print(train_df, target_data, crossval_train, crossval_target)


		learning_models = Learning_Models()

		learning_models.fit(train_df, target_data_train)

		ensemble_sm_accuracy,ensemble_prob_accuracy = learning_models.ensemble(train_df, target_data_train)

		#print(ensemble_sm_accuracy)
		#print("\n\n")
		#print(ensemble_prob_accuracy)

		#learning_models.test(test_df)

		ensemble_crossval_sm_acc, ensemble_crossval_prob_acc = learning_models.ensemble(crossval_train, crossval_target)

		#print(ensemble_crossval_sm_acc)
		#print("\n\n")
		#print(ensemble_crossval_prob_acc)

		
		print("Simple Majority Accuracies:")
		for model_sm in ensemble_sm_accuracy.keys():
			print("Model: "+str(model_sm)+"Train Accuracy: "+str(ensemble_sm_accuracy[model_sm])+" CrossVal Accuracy: "+str(ensemble_crossval_sm_acc[model_sm]))
		

		best_model_sm = []
		best_acc_heuristic_val_sm = -1 #Least Accuracy Value		
		for model_sm in ensemble_sm_accuracy.keys():
			current_heuristic_val = ensemble_sm_accuracy[model_sm] - abs(ensemble_sm_accuracy[model_sm] - ensemble_crossval_sm_acc[model_sm])
			
			if current_heuristic_val > best_acc_heuristic_val_sm:
				best_model_sm = model_sm
				best_acc_heuristic_val_sm = current_heuristic_val

		print("Best Model based on simple Majority: "+str(best_model_sm)+" Train Acc: "+str(ensemble_sm_accuracy[best_model_sm])+" Cross Val Accuracy: "+str(ensemble_crossval_sm_acc[best_model_sm]))

		'''
		print("Average Probability Accuracies:")
		for model_sm in ensemble_prob_accuracy.keys():
			print("Model: "+str(model_sm)+" Train Accuracy: "+str(ensemble_prob_accuracy[model_sm])+" CrossVal Accuracy: "+str(ensemble_crossval_prob_acc[model_sm]))
		'''

		best_model_prob = []
		best_acc_heuristic_val_prob = -1 #Least Accuracy Value		
		for model_prob in ensemble_prob_accuracy.keys():
			current_heuristic_val = ensemble_prob_accuracy[model_prob] - abs(ensemble_prob_accuracy[model_prob] - ensemble_crossval_prob_acc[model_prob])
			
			if current_heuristic_val > best_acc_heuristic_val_prob:
				best_model_prob = model_prob
				best_acc_heuristic_val_prob = current_heuristic_val

		print("Best Model based on Average Probability: "+str(best_model_prob)+" Train Acc: "+str(ensemble_prob_accuracy[best_model_prob])+" Cross Val Accuracy: "+str(ensemble_crossval_prob_acc[best_model_prob]))

		best_model_overall = []
		sm_or_prob = 0
		best_train_accuracy = -1
		best_cv_accuracy = -1

		if best_acc_heuristic_val_sm > best_acc_heuristic_val_prob:
			sm_or_prob = 1
			best_model_overall = best_model_sm
			best_train_accuracy = ensemble_sm_accuracy[best_model_overall]
			best_cv_accuracy = ensemble_crossval_sm_acc[best_model_overall]
		else:
			sm_or_prob = 2
			best_model_overall = best_model_prob
			best_train_accuracy = ensemble_prob_accuracy[best_model_overall]
			best_cv_accuracy = ensemble_crossval_prob_acc[best_model_overall]

		test_data_without_target = test_df.iloc[:, :-1]
		target_data_test = test_df.iloc[:,-1]

		ensemble_test_sm_acc, ensemble_test_prob_acc = learning_models.ensemble(test_data_without_target, target_data_test)
		
		best_test_accuracy = ensemble_test_sm_acc[best_model_overall] if sm_or_prob == 1 else ensemble_test_prob_acc[best_model_overall]

		if isinstance(best_model_overall, int) == False:
			best_method = "Simple Majority " if sm_or_prob == 1 else "Scored Probability "
		else:
			best_method = "Single Model "	

		#print("Best Ensemble Method: "+str(best_method)+"Best Model Overall:"+str(best_model_overall)+" Train Accuracy: "+str(best_train_accuracy)+" Cross Validation Accuracy: "+str(best_cv_accuracy)+" Test Accuracy: "+str(best_test_accuracy))

		return best_method, self.process_best_model_as_string(best_model_overall), best_train_accuracy, best_cv_accuracy, best_test_accuracy

	def process_best_model_as_string(self, best_model_overall):
		encode_models = {}
		encode_models[0] = "Decision Tree"
		encode_models[1] = "Naive Bayes"
		encode_models[2] = "Support Vector Machines"
		encode_models[3] = "Neural Network"
		encode_models[4] = "Logistic Regression"
		if isinstance(best_model_overall, int) == True:
			return encode_models[best_model_overall]

		result_str = ""
		for m in range(0, len(best_model_overall)-1):
			result_str += str(encode_models[best_model_overall[m]]) + ","
		result_str += str(encode_models[best_model_overall[len(best_model_overall)-1]])
		return result_str
		

def init_flow(train_folder, test_folder):
	result_list = []
	result_columns = ["Best Ensemble Method", "Best Machine Learning Method", "Training Accuracy", "Cross Validated Accuracy", "Test Set Accuracy"]
	for seed_no in range(1, 3):
		train_data_path = str(train_folder) + "Train" + str(seed_no) + ".csv"
		test_data_path = str(test_folder) + "Test" + str(seed_no) + ".csv"
	
		print("Running Seed Number: "+str(seed_no))
		best_method, best_model, train_acc, cv_acc, test_acc = Solution().fake_reviews_detection(train_data_path, test_data_path)
		result_list.append([best_method, best_model, train_acc, cv_acc, test_acc])
		print("---------------------------------------")

	return pd.DataFrame(result_list, columns = result_columns)

def run_liwc():

	print("Running LIWC-------------------------")

	result_folder = "../Results/"

	train_folder = "../Data/LIWC/Train/"
	test_folder = "../Data/LIWC/Test/"

	result_df_liwc = init_flow(train_folder, test_folder)

	result_df_liwc.to_csv(result_folder + "LIWC_Results.csv", index = False)

def run_pos():

	print("Running POS-------------------------")

	result_folder = "../Results/"

	train_folder = "../Data/POS/Train/"
	test_folder = "../Data/POS/Test/"

	result_df_pos = init_flow(train_folder, test_folder)

	result_df_pos.to_csv(result_folder + "POS_Results.csv", index = False)

if __name__ == "__main__":
	run_liwc()
	run_pos()
