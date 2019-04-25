# Fake_Reviews_Detection

This is a README for the Fake Reviews Detection project by Group 22.

Team Members:
Manthan Sanghavi
Supriyaa Damodaraswamy
Shreekrishna Prasad Sethuraman
Waris Phupaibhul
Vishnunarayanan Ramasubramanian

Notes:
1: The shell script 'Group22_Fake_Reviews.sh' invokes the core Machine Learning model and generates results.
2: Various pre-processing steps have been applied on the dataset given. The code and results for pre-processing are also included in this repository.
3: The learning models are located in "Fake_Reviews_Detection/Machine_Learning_Models/Code". The name of the python script is "learning_models.py".
4: The above python script can be invoked directly from the command line or through the bash script given.
5: The results of the program are generated in "Fake_Reviews_Detection/Machine_Learning_Models/Results". There are three .csv files generated.
6: The "all_results.csv" file contains evaluation metrics values, averaged over ten shuffles of data, for each of the pre-processing methods.
7: The "LIWC_Results.csv" and "POS_Results.csv" contain the best model picked per seed for both Linguistic Inquiry Word Count (LIWC) and Part-of-speech(POS) tagging on the data.
8: Since there are multiple layers of ensembling in the algorithm, the best model can be either a single model or an ensemble. This is also mentioned in the result file.
9: The two ensemble methods explored here in multiple layers are "simple majority" and "average scored probability". 
10: The datasets used for learning, after pre-processing are present in "Fake_Reviews_Detection/Machine_Learning_Models/Data". It also contains ten train-test splits of the data used. All of these seeds are run through the program.

Instructions:
To run the project, simply execute the "Group22_Fake_Reviews.sh" script from the command line.

System Requirements:
1: Ubuntu 16.04 or any Linux distribution.
2: Python 3.5.2.
3: All the required packages are installed as per "requirements.txt" with the correct version using pip3.
