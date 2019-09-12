'''
    Shreyash Shrivastava
    1001397477
    CSE 4309
'''
import numpy as np
import pandas as pd
import csv
# import matplotlib.pyplot as plt
# import seaborn as sns
import math
# import statistics
import os
import sys

testingfileName = sys.argv[2] 
trainingfileName = sys.argv[1] 



THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_file_training = os.path.join(THIS_FOLDER, trainingfileName)
my_file_testing= os.path.join(THIS_FOLDER, testingfileName)
# filename_training = sys.argv[1]
# filename_testing = sys.argv[2]


# my_file_training = 'pendigits_training.txt' 
# my_file_testing = 'pendigits_test.txt'

# Training 
data_list = []
index_range = 0

f = open(my_file_training)
for l in f.readlines():
    #print(l.strip().split('\t')[0])
    row = l.strip().split('\t')[0]
    add_row = row.split()
    add_row = [float(x) for x in add_row]
    data_list.append(add_row)
    index_range = len(row.split())
f.close()  

index_=[]

for x in range(1,index_range+1):
    index_.append(str(x))

df = pd.DataFrame(data_list,columns=index_)

df.columns= map(int,df.columns)
#Prior Probabilities of different classes
classes = list(df[df.columns[-1]].value_counts().index)
classes = [int(x) for x in classes]
classes_count = df[df.columns[-1]].value_counts()


# Prior Probability of classes
probabilities = []
for x in classes_count:
    probabilities.append(float(x/sum(classes_count)))
    
class_probabilities = {}
for x in range(len(classes_count)):
    class_probabilities.update({classes[x]:probabilities[x]})


# Functions for mean and standard deviation for each class
def mean(class_num,df):
    mean_nums = []
    for y in (df.columns[:-1]):
        col_list = []
        for x in range(len(df.index)):
            if (list(df.iloc[x])[-1]) == float(class_num):
                col_list.append(df[y][x])
        mean_nums.append(np.mean(col_list))
        
    return (class_num,mean_nums)


def standard_deviation(class_num,df):
    std_nums = []
    for y in (df.columns[:-1]):
        col_list = []
        for x in range(len(df.index)):
            if (list(df.iloc[x])[-1]) == float(class_num):
                col_list.append(df[y][x])
        if np.std(col_list) < float(0.01):
            std_nums.append(0.01)
        else:  
            std_nums.append(np.std(col_list))
               
    return (class_num,std_nums)

# Dictonary of mean and standard_deviation (Since we are assuming a Gaussian Distribution) 
# Dictonary structure : Dict(class:(class,[mean values of differnt attributes]))
classes_mean = {}
classes_std = {}

for x in classes:
    classes_mean.update({x:mean(x,df)})

for x in classes:
    classes_std.update({x:standard_deviation(x,df)})


# Making prediction 

def calcProb(x,mean,std):
    exp = np.exp(-(np.power(x-mean,2)/(2*np.power(std,2))))
    return (1/ (np.sqrt(2*np.pi)*std))*exp

# Multiplying attribute probabilites of for each class
def probabilityOfClass(class_num,test_vector):
    calculated_probability_of_testVector = []
    count = 0
    for x in test_vector:
        mean_class = classes_mean[class_num][1][count]
        mean_std = classes_std[class_num][1][count]
        prob = calcProb(x,mean_class,mean_std)
        count+=1
        calculated_probability_of_testVector.append(prob)
    
    #print(calculated_probability_of_testVector)
    ret = 1.0
    for x in calculated_probability_of_testVector:
        ret=ret*x
    
    return ret*class_probabilities[class_num]

#output training phase
print('### OUTPUT TRAINING PHASE ###')
print()
for x in sorted(classes_mean):
    attribute = 0
    for y in range(len(classes_mean[x][1:][0])):
        print('Class %d, attribute %d, mean = %.2f, std=%.2f' % (x,attribute+1,classes_mean[x][1][attribute],classes_std[x][1:][0][attribute]))
        #print(classes_mean[x][1:][0][attribute])
        attribute+=1
    print()


# Testing
data_list = []
index_range = 0
f = open(my_file_testing)
for l in f.readlines():
    #print(l.strip().split('\t')[0])
    row = l.strip().split('\t')[0]
    add_row = row.split()
    add_row = [float(x) for x in add_row]
    data_list.append(add_row)
    index_range = len(row.split())
f.close()  

df_test = pd.DataFrame(data_list,columns=index_)

#output testing phase
print('### OUTPUT TESTING PHASE ###')
print()
accuracy = 0
counter=1
accuracy_arr = []
for x in df_test.index:
    actual_class = ((int)((list(df_test.iloc[x])[-1])))
    test_vector = (list(df_test.iloc[x]))[:-1]
    #print(test_vector)
    
    classificationS = {}
    for x in classes:
        classificationS.update({x:probabilityOfClass(x,test_vector)})
    accu = 0
    
    if max(classificationS, key=classificationS.get) == actual_class:     
        tie_count = 0
        for x in (classificationS):
            if classificationS[max(classificationS,key=classificationS.get)] == classificationS[x]:
            #print(classificationS[max(classificationS,key=classificationS.get)])
                tie_count+=1
                
        if tie_count>1:
            accuracy_arr.append(1/tie_count) 
            accu = 1/tie_count
        else:       
            accuracy+=1
            accu=1
            accuracy_arr.append(accu)
    else:
        accuracy_arr.append(0.0)
    print('ID=%5d, predicted=%3d, probability = %.4f, true=%3d, accuracy=%4.2f' %(counter,max(classificationS, key=classificationS.get),
                                                                                 classificationS[max(classificationS, key=classificationS.get)],
                                                                                 actual_class,accu))
    counter+=1

print()
print('classification accuracy =%6.4f' % (sum(accuracy_arr)/len(df_test)))

# # Sklearn NB
# from sklearn import datasets
# iris = datasets.load_iris()
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (iris.data.shape[0],(iris.target != y_pred).sum()))


# # In[ ]:


# y_pre = gnb.fit(df,)


# # In[ ]:


# iris.data


# # In[28]:


# predict_actual= list(df[9])
# predict_actual = [int(x) for x in predict_actual]
# predict_actual


# # In[29]:


# predict= list(df_test['9'])
# predict = [int(x) for x in predict]
# predict


# # In[30]:


# df.iloc[:, 0:8]


# # In[31]:


# y_pred = gnb.fit(df.iloc[:, 0:8], predict_actual).predict(df_test.iloc[:, 0:8])


# # In[32]:


# count=0
# for x in range(len(y_pred)):
#     if y_pred[x] == predict[x]:
#         count+=1
# count/(len(df_test.index))

