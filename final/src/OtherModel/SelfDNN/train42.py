import csv
import pickle as pk
import csv
import numpy as np
import scipy as sp
import sys
from keras.models import Model, Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2
from keras.models import model_from_json
from sklearn import preprocessing

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
print label1,len(label1)
print label2,len(label2)
print label3,len(label3)
print label4,len(label4)

model=Sequential()

model.add(Dense(128,input_shape=(41,)))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.3)) 
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.3)) 
model.add(Dropout(0.25))
model.add(Dense(512))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.3)) 
model.add(Dropout(0.25))
model.add(Dense(len(label4)))
model.add(Activation('softmax'))
model.summary()

#adam=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['fmeasure'])
early_stopping = EarlyStopping(monitor='val_fmeasure', patience=1)

train_set_input=csv.reader(open('train','rb'))
data = []
index = 0
n = 400000
for it in range (0,1):
    train_set=[]
    train_y=[]
    for row in train_set_input:
        row[1]=label1.index(row[1])
        row[2]=label2.index(row[2])
        row[3]=label3.index(row[3])
        y_class = label_list[type_list.index(row[41])]
        row[41]=y_class 
               
        data.append(np.array(row))
        index += 1
        if index % n == 0 and index / n == 1:
            DATA = np.array(data)
            data = []
            print DATA.shape
        elif index % n == 0:
            DATA = np.vstack((DATA,np.array(data)))
            data = []
            print DATA.shape
        
    DATA = np.vstack((DATA,np.array(data))).astype(float)       
    print DATA.shape
    print len(label1),len(label2),len(label3)
    print 'extra len',len(label1)+len(label2)+len(label3)
    normalizer = preprocessing.Normalizer().fit(DATA[:,0:41])
    DATA[:,0:41] = normalizer.transform(DATA[:,0:41])
    class_weight = {0:4.,1:1.,2:81649.,3:3600.,4:99.}
    model.fit(DATA[:,0:41],DATA[:,41],batch_size=512,nb_epoch=10,validation_split=0.01,class_weight = [4,1,90000,4000,300])

model.save('model_099.h5')
test_set=[]
test_file=csv.reader(open('test.in','rb'))
for row in test_file:
    if row[2]=='icmp':
        row[2]='http'
    row[1]=label1.index(row[1])
    row[2]=label2.index(row[2])
    row[3]=label3.index(row[3])
    test_set.append(map(float,row))
test_set = normalizer.transform(test_set)
prediction=model.predict(test_set,verbose=1)

for it in range (0,10):
    test_val = []
    test_val_y = []
    count = 0
    for i in range(prediction.shape[0]):
        max_index = np.argmax(prediction[i])
        if prediction[i][max_index] >= 0.9999 or (prediction[i][max_index] >= 0.99 and max_index != 0 and max_index != 1):
            count+=1
            test_val.append(test_set[i])
            test_val_y.append([max_index])
    print count,'of probe'
    if len(test_val) != 0 :
        print 'have probe',it
        test_val = np.vstack((DATA[:,0:41],np.array(test_val)))
        test_val_y = np.concatenate((DATA[:,41],np.array(test_val_y).reshape((-1,))),axis = 0)
        model.fit(test_val,test_val_y,batch_size=512,nb_epoch=3,class_weight = [4,1,90000,4000,200],shuffle=True)
        prediction=model.predict(test_set,verbose=1)
    else :
        print 'no probe',it
#model.fit(DATA[:,0:41],DATA[:,41],batch_size=512,nb_epoch=10,validation_split=0.01,class_weight = [4,1,84649,3600,99],callbacks=[early_stopping])
model.save('model2_099.h5')
