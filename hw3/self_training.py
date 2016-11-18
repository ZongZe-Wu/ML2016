import numpy as np
import sys
from PIL import Image
from matplotlib import pyplot as plt
from scipy.misc import toimage 
from keras.models import Sequential
from keras.layers.core import MaxoutDense,Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2
from keras.models import load_model
import pickle


def main(argv):
	all_label = pickle.load(open(argv[0]+'all_label.p','rb'))
	all_unlabel = pickle.load(open(argv[0]+'all_unlabel.p','rb'))
	#label_test =  pickle.load(open(argv[0]+'test.p','rb'))
	all_label = np.array(all_label)
	#label_test = np.array(label_test['data'])
	all_unlabel = np.array(all_unlabel)

	all_label = all_label.reshape(5000,3,32,32)
	all_unlabel = all_unlabel.reshape(45000,3,32,32)
	#label_test = label_test.reshape(10000,3,32,32)

	x_train = all_label.astype('float32')/255
	x_unlabel = all_unlabel.astype('float32')/255
	#x_test = label_test.astype('float32')/255
	print all_label.shape
	print all_unlabel.shape
	#print label_test.shape
	all_label_y_train = []
	label = -1
	for i in range (5000):
		if i % 500 == 0 :
			label=label+1
		all_label_y_train.append(label)
		
	all_label_y_train = np.array(all_label_y_train)
	check_list = [False for i in range(45000)]
	# convert class vectors to binary class matrices
	for i in range (15):
	
		model = Sequential()
		model.add(Convolution2D(32,3,3,input_shape=(3,32,32),activation='relu',dim_ordering='th',border_mode='same'))	
		model.add(Dropout(0.2))
		model.add(Convolution2D(32, 3, 3,dim_ordering='th',activation='relu',border_mode='same'))
		model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))
		model.add(Convolution2D(64, 3, 3,dim_ordering='th',activation='relu',border_mode='same'))
		model.add(Dropout(0.2))
		model.add(Convolution2D(64, 3, 3,dim_ordering='th',activation='relu',border_mode='same'))
		model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))
		model.add(Convolution2D(128, 3, 3,dim_ordering='th',activation='relu',border_mode='same'))
		model.add(Dropout(0.2))
		model.add(Convolution2D(128, 3, 3,dim_ordering='th',activation='relu',border_mode='same'))
		model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))	

		model.add(Flatten())
		model.add(Dropout(0.25))
		model.add(Dense(1024))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dropout(0.25))
		model.add(Dense(512))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dropout(0.25))
		model.add(Dense(10))
		model.add(Activation('softmax'))

		model.summary()

		model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
		y_train = np_utils.to_categorical(all_label_y_train, 10)
		model.fit(x_train,y_train,batch_size=100,nb_epoch=50)	
			
		print 'finish_epoch'
		unlabel_test = model.predict(all_unlabel)
		print 'finish_predict'		
		for j in range (i*3000,(i+1)*3000):
			if np.amax(unlabel_test[j]) >=0.8 and check_list[j] == False:
				x_train=np.vstack((x_train,all_unlabel[j].reshape((1,3,32,32))))
				all_label_y_train=np.append(all_label_y_train,np.argmax(unlabel_test[j]))
				check_list[j] = True
			if j%500==0:
				print j
		if i ==14:
			y_train = np_utils.to_categorical(all_label_y_train, 10)
			model.fit(x_train,y_train,batch_size=100,nb_epoch=50)	
	
		
	model.save_weights(argv[1]+'.hdf5')
	archi = model.to_json()
	file = open(argv[1]+'.json','w')
	file.write(archi)
	''''	
	y_test = model.predict_classes(x_test)
	file = open('test.csv','w')
	file.write('ID,class\n')
	for i in range(10000):
		file.write('{0}'.format(i))
		file.write(',')
		file.write(str(y_test[i]))
		file.write('\n')
	file.close()	
	'''''
if __name__ == '__main__':
	main(sys.argv[1:])
