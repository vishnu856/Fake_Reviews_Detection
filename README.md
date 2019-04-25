# Fake_Reviews_Detection

This is a README for the Fake Reviews Detection project by Group 22.

## Team Members:
* Manthan Sanghavi
* Supriyaa Damodaraswamy
* Shreekrishna Prasad Sethuraman
* Waris Phupaibhul
* Vishnunarayanan Ramasubramanian

## Notes:
* The shell script 'Group22_Fake_Reviews.sh' invokes the core Machine Learning model and generates results.
* Various pre-processing steps have been applied on the dataset given. The code and results for pre-processing are also included in this repository.
* The learning models are located in "Fake_Reviews_Detection/Machine_Learning_Models/Code". The name of the python script is "learning_models.py".
* The above python script can be invoked directly from the command line or through the bash script given.
* The results of the program are generated in "Fake_Reviews_Detection/Machine_Learning_Models/Results". There are three .csv files generated.
* The "all_results.csv" file contains evaluation metrics values, averaged over ten shuffles of data, for each of the pre-> processing methods.
* The "LIWC_Results.csv" and "POS_Results.csv" contain the best model picked per seed for both Linguistic Inquiry Word Count (LIWC) and Part-of-speech(POS) tagging on the data.
* Since there are multiple layers of ensembling in the algorithm, the best model can be either a single model or an ensemble. This is also mentioned in the result file.
* The two ensemble methods explored here in multiple layers are "simple majority" and "average scored probability". 
* The datasets used for learning, after pre-processing are present in "Fake_Reviews_Detection/Machine_Learning_Models/Data". It also contains ten train-test splits of the data used. All of these seeds are run through the program.

## Instructions:
* To run the project, simply execute the "Group22_Fake_Reviews.sh" script from the command line.
* Example:
* `root@commandline$bash Group22_Fake_Reviews.sh`

## System Requirements:
* Ubuntu 16.04 or any Linux distribution.
* Python 3.5.2.
* All the required packages are installed as per "requirements.txt" with the correct version using pip3.
