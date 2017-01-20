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

model=Sequential()
model.add(Dense(128,input_shape=(41,)))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.3)) 
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.3)) 
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(BatchNormalization())
#model.add(Activation('relu'))
model.add(LeakyReLU(alpha=0.3)) 
model.add(Dropout(0.25))
model.add(Dense(len(label4)))
model.add(Activation('softmax'))
model.summary()
adam=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='sparse_categorical_crossentropy',optimizer=adam,metrics=['fmeasure'])
early_stopping = EarlyStopping(monitor='val_fmeasure', patience=1)

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
        data.append(np.array(row))
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
    model.fit(DATA[:,0:41],DATA[:,41],batch_size=32,nb_epoch=10,shuffle=True)

model.save('model.h5')



