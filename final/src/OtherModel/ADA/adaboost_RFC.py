import csv
import pickle as pk
import csv
import numpy as np
import scipy as sp
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
'''=======================
     preprocessing
======================='''
type_list=['normal.']
label_list=[0]
with open('training_attack_types.txt','r') as attack_type:
    for line in attack_type:
        line=line.strip().split(' ')
        if line[1]=='dos':
            type_list.append(line[0]+'.')
            label_list.append(1)
        elif line[1]=='u2r':
            type_list.append(line[0]+'.')
            label_list.append(2)
        elif line[1]=='r2l':
            type_list.append(line[0]+'.')
            label_list.append(3)
        elif line[1]=='probe':
            type_list.append(line[0]+'.')
            label_list.append(4)
print type_list
print label_list 


label1=[]
label2=[]
label3=[]
label4=['normal','dos','u2r','r2l','porbe'] # just use the len information
train_set=[]

train_set_input=csv.reader(open('train','rb'))
for row in train_set_input:
    if row[1] not in label1:
        label1.append(row[1])
    if row[2] not in label2:
        label2.append(row[2])
    if row[3] not in label3:
        label3.append(row[3])
#    print row[1:4],row[41]
print label1
print label2
print label3
print label4

'''=======================
     load data
======================='''

train_set_input=csv.reader(open('train','rb'))
data = []
index = 0
n=50*10000
for it in range (0,1):
    train_set=[]
    train_y=[]
    for row in train_set_input:
        row[1]=label1.index(row[1])
        row[2]=label2.index(row[2])
        row[3]=label3.index(row[3])
        y_class = label_list[type_list.index(row[41])]
        row[41]=y_class
        data.append(np.array(row).astype(float))
        index += 1
        if index % n == 0 and index / n == 1:
            DATA = np.array(data)
            data = []
            print index 
        elif index % n == 0:
            DATA = np.vstack((DATA,np.array(data)))
            data = []
            print index
    DATA = np.vstack((DATA,np.array(data)))       
    print DATA.shape
    print DATA[5]

'''=======================
     RFC+adaboost
======================='''

RFC = RandomForestClassifier(n_estimators=20,verbose=1)
adaboost_classfier = AdaBoostClassifier(RFC,n_estimators=500,learning_rate=1)
adaboost_classfier.fit(DATA[:,0:41],DATA[:,41].astype(int))
DATA=[]

'''=======================
     predict
======================='''

i=0
test_set=[]
test_file=csv.reader(open('test.in','rb'))
for row in test_file:
    i+=1
    if row[2]=='icmp':
        row[2]='http'
    row[1]=label1.index(row[1])
    row[2]=label2.index(row[2])
    row[3]=label3.index(row[3])
    test_set.append(map(float,row))

prediction=adaboost_classfier.predict(test_set)
#for i in range(len(prediction)):
    #label_string=label4[prediction[i]]
    #index=type_list.index(label_string)
    #prediction[i]=label_list[index]
print prediction

count = np.zeros(5)
i=1
prediction_file=open ('prediction_adab_RFC0106.csv','w')
prediction_file.write('id,label\n')
for label in prediction:
    prediction_file.write(str(i)+','+str(label)+'\n')
    count[label] += 1
    i+=1
print count

