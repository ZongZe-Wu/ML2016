#!/usr/bin/python
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
from sklearn import ensemble
from keras.models import load_model
from numpy import genfromtxt
def CreatModel(input_size):
  model=Sequential()
  model.add(Dense(input_size,input_shape=(input_size,)))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.3))
  model.add(Dense(4))
  model.add(Dropout(0.3))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.3))
  model.add(Dense(2))
  model.add(Dropout(0.3))
  model.add(BatchNormalization())
  model.add(LeakyReLU(alpha=0.3))
  model.add(Dense(output_dim=1))
  model.add(Activation('sigmoid'))
  #adam=Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
  return model
def NormalizeClassesData(data_X):#data_X:list of 2D numpy array
  all_data=data_X[0]
  for i in range(1,len(data_X)):
    all_data = np.vstack((all_data,data_X[i]))
    print 'stack to',i
  normalizer = preprocessing.StandardScaler()
  normalizer.fit(all_data.astype(float))
  all_data=[]
  for i in range(len(data_X)):
    data_X[i] = normalizer.transform(data_X[i].astype(float))
  print 'normalization,ex 0:',np.shape(data_X[0])
  return [data_X,normalizer]
def ReduceClassesFeature(class_id1,class_id2,data_X_1,data_X_2,importance_treshold):#two 2D numpy array and corresponding class id
  X = np.vstack((data_X_1,data_X_2))
  y = np.asarray( [class_id1]*len(data_X_1)+[class_id2]*len(data_X_2) )
  ET_classfier = ensemble.ExtraTreesClassifier()
  ET_classfier.fit(X,y)
  importance = ET_classfier.feature_importances_
  feature_select_mask = (importance>importance_treshold)
  X = []
  y=[]
  data_X_1 = data_X_1[:,feature_select_mask]
  data_X_2 = data_X_2[:,feature_select_mask]  
  remainder_ratio = 1
  if class_id1 <2 and class_id2 > 1:#class_id1 = 0,1 class_id2 = 2,3,4
    #very imbalance data
    #sample the big set    
    rand_idx = np.random.permutation(len(data_X_1))
    data_X_1 = data_X_1[rand_idx[0:remainder_ratio*len(data_X_2)],:]                 
  elif class_id2 <2 and class_id1 > 1: #class_id2 = 0,1 class_id1 = 2,3,4 (not happend in the one vs one models)
    rand_idx = np.random.permutation(len(data_X_2))
    data_X_2 = data_X_2[rand_idx[0:remainder_ratio*len(data_X_1)],:]
  elif class_id1==0 and class_id2==1:
    rand_idx = np.random.permutation(len(data_X_1))
    data_X_2 = data_X_2[rand_idx[0:remainder_ratio*len(data_X_1)],:]
      
  print 'reduct shape',np.shape(data_X_1)
  print 'mask',feature_select_mask
  return [data_X_1,data_X_2,ET_classfier]
def main():
  print '[main]1 vs 1,five class'
  importance_treshold = 0.01
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
  train_set_input=csv.reader(open('train','rb'))
  data = []
  index = 0
  n = 400*1000
  data_X=[[],[],[],[],[]]
  tmp_X_0 = []
  tmp_X_1 = []
  tmp_X_2 = []
  tmp_X_3 = []
  tmp_X_4 = []
  for row in train_set_input:
    row[1]=label1.index(row[1])
    row[2]=label2.index(row[2])
    row[3]=label3.index(row[3])
    y_class = label_list[type_list.index(row[41])]#5 class
    row[41]=y_class 
    if y_class == 0:
      tmp_X_0.append(row[0:41])
    elif y_class==1:
      tmp_X_1.append(row[0:41])
    elif y_class==2:
      tmp_X_2.append(row[0:41])
    elif y_class==3:
      tmp_X_3.append(row[0:41])
    elif y_class==4:
      tmp_X_4.append(row[0:41])         
    index +=1
    if index % n == 0 and index / n == 1:#first chunk
      data_X[0] = np.array(tmp_X_0)
      data_X[1] = np.array(tmp_X_1)
      data_X[2] = np.array(tmp_X_2)
      data_X[3] = np.array(tmp_X_3)
      data_X[4] = np.array(tmp_X_4)
      tmp_X_0 = []
      tmp_X_1 = []
      tmp_X_2 = []
      tmp_X_3 = []
      tmp_X_4 = [] 
    elif index % n == 0 : #other chunk
      if len(tmp_X_0)>0:
        data_X[0] = np.vstack((data_X[0],np.array(tmp_X_0)))
      if len(tmp_X_1)>0:
        data_X[1] = np.vstack((data_X[1],np.array(tmp_X_1)))  
      if len(tmp_X_2)>0:
        data_X[2] = np.vstack((data_X[2],np.array(tmp_X_2)))  
      if len(tmp_X_3)>0:
        data_X[3] = np.vstack((data_X[3],np.array(tmp_X_3)))  
      if len(tmp_X_4)>0:
        data_X[4] = np.vstack((data_X[4],np.array(tmp_X_4)))   
      tmp_X_0 = []
      tmp_X_1 = []
      tmp_X_2 = []
      tmp_X_3 = []
      tmp_X_4 = []
  #for the rest data
  if len(tmp_X_0)>0:
    data_X[0] = np.vstack((data_X[0],np.array(tmp_X_0)))
  if len(tmp_X_1)>0:
    data_X[1] = np.vstack((data_X[1],np.array(tmp_X_1)))  
  if len(tmp_X_2)>0:
    data_X[2] = np.vstack((data_X[2],np.array(tmp_X_2)))  
  if len(tmp_X_3)>0:
    data_X[3] = np.vstack((data_X[3],np.array(tmp_X_3)))  
  if len(tmp_X_4)>0:
    data_X[4] = np.vstack((data_X[4],np.array(tmp_X_4)))   
  tmp_X_0 = []
  tmp_X_1 = []
  tmp_X_2 = []
  tmp_X_3 = []
  tmp_X_4 = [] 
  print '[main] data shpae = ',np.shape(data_X)
  print '[main] data shape 0 = ',np.shape(data_X[0])
  print '[main] data shape 1 = ',np.shape(data_X[1])
  print '[main] data shape 2 = ',np.shape(data_X[2])
  print '[main] data shape 3 = ',np.shape(data_X[3])
  print '[main] data shape 4 = ',np.shape(data_X[4])
  #normalize over all training data
  [data_X,normalizer] = NormalizeClassesData(data_X)
  one_to_one_list=[]#each element =[i,j,ETF_classfy,model],i is positive ,j is negative
  #for each pair of class
  feature_reduce = genfromtxt('feature_reduce.out', delimiter=',') 
  tmp_idx = 0 
  for i in range(len(data_X)):
    for j in range(i+1,len(data_X)):   
      print '[main] do reduction feature'
      #[train_positive,train_negative,ET_classfier]=ReduceClassesFeature(i,j,data_X[i],data_X[j],importance_treshold)      
      #print '[main] load a binary model for',i,',',j,'size',np.shape(train_positive),np.shape(train_negative)
      #model = CreatModel(np.shape(train_positive)[1])
      print '[main] train the binary model'
      '''
      train_X = np.vstack((train_positive,train_negative))
      train_Y = np.asarray( [0]*len(train_positive)+[1]*len(train_negative) )
      rand_idx = np.random.permutation(len(train_X))
      train_X = train_X[rand_idx]
      train_Y = train_Y[rand_idx]
      if j>1:
        batch = 8
        epoch = 25
      else:
        batch = 32
        epoch = 10
      early_stopping = EarlyStopping(monitor='val_acc', patience=5)
      #model.fit(train_X.astype(float),train_Y.astype(float),batch_size=batch,nb_epoch=epoch,shuffle=True,validation_split=0.1,callbacks=[early_stopping])
      '''
      model = load_model('model'+str(i)+str(j)+'.h5')
      model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
      print '[main] save one_to_one_list'
      one_to_one_list.append([i,j,feature_reduce[tmp_idx],model])
      tmp_idx +=1
  data_X=[]
  model=[]
  #testing
  print '---testing---'
  #raw_input('press any key to continue')
  test_set=[]
  test_file=csv.reader(open('test.in','rb'))
  for row in test_file:    
    if row[2]=='icmp':
        row[2]='http'
    row[1]=label1.index(row[1])
    row[2]=label2.index(row[2])
    row[3]=label3.index(row[3])    
    test_set.append(map(float,row))
  test_set = np.asarray(test_set)
  test_set = normalizer.transform(test_set)#use the same normalizer model as training set
  print 'test shape = ',test_set.shape 
  test_score = np.zeros((np.shape(test_set)[0],5)) 
  #pass to each model
  for it in range(len(one_to_one_list)):
    model = []
    [i,j,feature_select_mask,model] = one_to_one_list[it]
    #transform by ETC
    test_X = test_set[:,feature_select_mask.astype(np.bool_)]#we need the same test set for all different model,so copy the data
    #get the score
    class_prob = model.predict_classes(test_X)
    class_prob = np.asarray(class_prob).reshape(-1)
    print '123123123:',test_X.shape,test_score[:,i].shape
    #print 'predict shape',np.shape(class_probe)
    print 'struct',i,j
    print 'score shape',np.shape(test_score[:,i])
    #save the score
    test_score[:,i] += 1- class_prob
    test_score[:,j] += class_prob
  #assign test data to the highest score class
  test_set =[]
  result = np.argmax(test_score,axis=1)
  test_score=[]
  #write to file
  i=1
  prediction_file=open ('prediction_1vs1.csv','w')
  prediction_file.write('id,label\n')
  for label in result:
    prediction_file.write(str(i)+','+str(label)+'\n')
    i+=1
  print 'finish'
if __name__ == '__main__':
  main()
