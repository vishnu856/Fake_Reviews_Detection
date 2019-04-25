import nltk
import os
import numpy as np
import csv

tag_list = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
            "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR",
            "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP",
            "VBZ", "WDT", "WP", "WP$", "WRB"]

folder_path = "C:\WaritCola\ASU\Spring2019\CSE 573 - Semantic Web Mining\Project\op_spam_v1.4"
polarity_list = ["positive_polarity", "negative_polarity"]
fold_list = ["fold1", "fold2", "fold3", "fold4"]
pos_source_list = ["truthful_from_TripAdvisor", "deceptive_from_MTurk"]
neg_source_list = ["truthful_from_Web", "deceptive_from_MTurk"]


# train = fold 1-4, test = fold 5
train_path1 = "C:\WaritCola\ASU\Spring2019\CSE 573 - Semantic Web Mining\Project\op_spam_v1.4\proj_training_pos"
train_path2 = "C:\WaritCola\ASU\Spring2019\CSE 573 - Semantic Web Mining\Project\op_spam_v1.4\proj_training_neg"
test_path1 = "C:\WaritCola\ASU\Spring2019\CSE 573 - Semantic Web Mining\Project\op_spam_v1.4\proj_testing_pos"
test_path2 = "C:\WaritCola\ASU\Spring2019\CSE 573 - Semantic Web Mining\Project\op_spam_v1.4\proj_testing_neg"


'''
def count_tags(current_count, input_list):
    row_data = [0] * (len(tag_list)+1)
    row_data[0] = current_count     # this is the id of each review
    for each_tag in input_list:
        try:
            i = tag_list.index(each_tag[1])
            row_data[i+1] += 1
        except ValueError:
            pass
    print(row_data)
    return row_data



# simple words to POS
comm = input("Enter 1 to do word to POS only (non-count)")
if comm == "1":
    fold_path = "C:\WaritCola\ASU\Spring2019\CSE 573 - Semantic Web Mining\Project\op_spam_v1.4.1"
    fold_list = ["fold1", "fold2", "fold3", "fold4", "fold5"]
    # create dataset with all features

    id_count = 1
    for sub_path in fold_list:
        train_data = []
        path = os.path.join(fold_path, sub_path)
        for each_review in os.listdir(path):
            f = open(os.path.join(path, each_review), 'r')
            review = f.read()
            tokens = nltk.word_tokenize(review)
            tagged = nltk.pos_tag(tokens)
            data_row = [id_count]
            if each_review[0] == "d":
                data_row.append(0)
            else:
                data_row.append(1)
            for each_tag in tagged:
                data_row.append(each_tag[1])
            train_data.append(data_row)
            id_count += 1

        # save to csv
        with open("{0}_pos.csv".format(sub_path), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(train_data)

    exit()
'''

# create dataset with all features
onehot = ["ID", "POSITIVE", "NEGATIVE", "TRUTHFUL", "DECEPTIVE", "MTURK", "WEB", "TRIPADVISOR"]
onehot_data = []
train_feature = []
train_data = []
id_count = 1
polarity = 0
source = 0
for a_pol in polarity_list:

    if polarity == 0:
        sl = pos_source_list
    else:
        sl = neg_source_list

    for a_source in sl:

        for a_fold in fold_list:

            path = folder_path+"\\"+a_pol+"\\"+a_source+"\\"+a_fold
            for each_review in os.listdir(path):

                # ---------------------------------------------------
                # ID + one hot encoding here
                onehotValue = [0] * len(onehot)
                onehotValue[0] = id_count
                if polarity == 0:  # positive polarity
                    onehotValue[1] = 1
                else:
                    onehotValue[2] = 1

                if each_review[0] == "d":   # deceptive
                    onehotValue[4] = 1
                else:
                    onehotValue[3] = 1

                # extract review source here
                if source == 0 and polarity == 0:
                    onehotValue[7] = 1
                elif source == 1 and polarity == 0:
                    onehotValue[6] = 1
                else:
                    onehotValue[5] = 1

                # extract name here
                a = each_review.find('_')
                b = each_review.rfind('_')
                hotel_name = each_review[a+1:b].upper()

                if hotel_name not in onehot:
                    onehot.append(hotel_name)
                    onehotValue.append(1)
                else:
                    i = onehot.index(hotel_name)
                    onehotValue[i] += 1

                # end of one hot section
                # ---------------------------------------------------

                f = open(os.path.join(path, each_review), 'r')
                review = f.read()
                tokens = nltk.word_tokenize(review)
                tagged = nltk.pos_tag(tokens)

                row_data = [0] * len(train_feature)
                for a_tag in tagged:
                    if a_tag[1] in train_feature:
                        i = train_feature.index(a_tag[1])
                        row_data[i] += 1
                    else:
                        train_feature.append(a_tag[1])
                        row_data.append(1)

                onehot_data.append(onehotValue)
                train_data.append(row_data)
                id_count += 1

        source = 1

    polarity = 1   # switch to negative polarity

# make sure that all data has the same element
for a_data in train_data:
    index_diff = len(train_feature)-len(a_data)
    if index_diff > 0:
        for i in range(index_diff):
            a_data.append(0)

for a_data in onehot_data:
    index_diff = len(onehot) - len(a_data)
    if index_diff > 0:
        for i in range(index_diff):
            a_data.append(0)

# do the same thing for testing (20%) data, but features will not be determined
# from these data
other_data = []     # holding the 'other' feature for all test data
onehot_data2 = []
test_data = []
polarity = 0
source = 0
for a_pol in polarity_list:

    if polarity == 0:
        sl = pos_source_list
    else:
        sl = neg_source_list

    for a_source in sl:

        path = folder_path + "\\" + a_pol + "\\" + a_source + "\\" + "fold5"
        for each_review in os.listdir(path):

            other = 0

            # ---------------------------------------------------
            # ID + one hot encoding here
            onehotValue = [0] * len(onehot)
            onehotValue[0] = id_count
            if polarity == 0:  # positive polarity
                onehotValue[1] = 1
            else:
                onehotValue[2] = 1

            if each_review[0] == "d":  # deceptive
                onehotValue[4] = 1
            else:
                onehotValue[3] = 1

            # extract review source here
            if source == 0 and polarity == 0:
                onehotValue[7] = 1
            elif source == 1 and polarity == 0:
                onehotValue[6] = 1
            else:
                onehotValue[5] = 1

            # extract name here (
            a = each_review.find('_')
            b = each_review.rfind('_')
            hotel_name = each_review[a + 1:b].upper()

            if hotel_name not in onehot:
                other += 1
            else:
                i = onehot.index(hotel_name)
                onehotValue[i] += 1

            # end of one hot section
            # ---------------------------------------------------

            f = open(os.path.join(path, each_review), 'r')
            review = f.read()
            tokens = nltk.word_tokenize(review)
            tagged = nltk.pos_tag(tokens)

            row_data = [0] * len(train_feature)
            for a_tag in tagged:
                if a_tag[1] in train_feature:
                    i = train_feature.index(a_tag[1])
                    row_data[i] += 1
                else:   # test data find POS tag not in training data's POS
                    other += 1

            onehot_data2.append(onehotValue)
            test_data.append(row_data)
            other_data.append(other)
            id_count += 1

        source = 1

    polarity = 1

# since test data uses feature from training data, all data will have the same features
# no need to pad zeros


# determine top 20 features
train_data_np = np.array(train_data)
count_np = np.sum(train_data_np, axis=0)
order_np = np.argsort(-count_np)
print(train_feature)
print(count_np)

# check correctness

# sum all non-top20 column
column_nottop20_train = train_data_np[:, order_np[19]]      # column 20
j = 20      # start on column 21
for i in range(len(order_np)-20):
    temp = train_data_np[:, order_np[j]]
    column_nottop20_train = column_nottop20_train + temp
    j += 1

# do the same for testing data
test_data_np = np.array(test_data)
column_nottop20_test = test_data_np[:, order_np[19]]      # column 20
j = 20      # start on column 21
for i in range(len(order_np)-20):
    temp = test_data_np[:, order_np[j]]
    column_nottop20_test = column_nottop20_test + temp
    j += 1

# remove non-top20 column of train/test data
# sorting index in descending order, so that index won't shift when deleting
remove_index = order_np[20:]
remove_index = -np.sort(-remove_index)      # descending sort
train_feature_final = train_feature
for index in remove_index:
    train_data_np = np.delete(train_data_np, index, 1)
    test_data_np = np.delete(test_data_np, index, 1)
    del train_feature_final[index]          # this handle column header

# add "not top 20" feature column and "other" column for test only
train_data_np = np.column_stack((np.array(onehot_data), train_data_np))
train_data_np = np.column_stack((train_data_np, column_nottop20_train))
test_data_np = np.column_stack((np.array(onehot_data2), test_data_np))
test_data_np = np.column_stack((test_data_np, column_nottop20_test))
test_data_np = np.column_stack((test_data_np, other_data))

# handle column header for train and test data
train_feature_final = onehot + train_feature_final
train_feature_final.append("NOT_TOP_20")
test_feature_final = train_feature_final.copy()
test_feature_final.append("OTHER")
header1 = str(train_feature_final)
header2 = str(test_feature_final)
remove_char_list = [" ", "[", "]", "'"]
for a_char in remove_char_list:
    header1 = header1.replace(a_char, "")
    header2 = header2.replace(a_char, "")
header1 = header1.replace(",,,", ",COMMA,")
header2 = header2.replace(",,,", ",COMMA,")

# everything is done, so let's save
fmt1 = ",".join(["%s"] + ["%d"] * (train_data_np.shape[1]-1))
fmt2 = ",".join(["%s"] + ["%d"] * (test_data_np.shape[1]-1))
np.savetxt("train.csv", train_data_np, fmt=fmt1, header=header1, comments='')
np.savetxt("test.csv", test_data_np, fmt=fmt2, header=header2, comments='')