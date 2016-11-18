import numpy as np
import sys
from PIL import Image
from matplotlib import pyplot as plt
from scipy.misc import toimage 
from keras.models import Sequential,Model
from keras.layers.core import MaxoutDense,Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten,UpSampling2D,Input
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1,l2,l1l2
from sklearn.cluster import KMeans
import pickle

def main(argv):
	all_label = pickle.load(open(argv[0]+'all_label.p','rb'))
	all_unlabel = pickle.load(open(argv[0]+'all_unlabel.p','rb'))
	#label_test =  pickle.load(open('test.p','rb'))
	all_label = np.array(all_label)
	#label_test = np.array(label_test['data'])
	all_unlabel = np.array(all_unlabel)

	all_label = all_label.reshape(5000,3,32,32)
	all_unlabel = all_unlabel.reshape(45000,3,32,32)
	#label_test = label_test.reshape(10000,3,32,32)
	x_train = all_label.astype('float32')/255
	x_unlabel = all_unlabel.astype('float32')/255
	#x_test = label_test.astype('float32')/255
	#toimage(x_unlabel[0]).show()
	input_img = Input(x_train[0].shape)

	x = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(input_img)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='th')(x)
	x = Convolution2D(16, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(x)
	x = MaxPooling2D((2, 2), border_mode='same', dim_ordering='th')(x)
	x = Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(x)
	encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering='th')(x)

	# at this point the representation is (16, 4, 4) i.e. 128-dimensional

	x = Convolution2D(8, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(encoded)
	x = UpSampling2D((2, 2), dim_ordering='th')(x)
	x = Convolution2D(16, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(x)
	x = UpSampling2D((2, 2), dim_ordering='th')(x)
	x = Convolution2D(32, 3, 3, activation='relu', dim_ordering='th',border_mode='same')(x)
	x = UpSampling2D((2, 2), dim_ordering='th')(x)

	decoded = Convolution2D(3, 3, 3, activation='sigmoid', dim_ordering='th',border_mode='same')(x)
	autoencoder = Model(input_img, decoded)
	autoencoder.summary()
	autoencoder.compile(optimizer='adam', loss='mse')
	autoencoder.fit(x_train, x_train,nb_epoch=100,batch_size=100,shuffle=True)	
	autoencoder.save_weights('auto.hdf5')
	auto = autoencoder.to_json()
	file = open('auto.json','w')
	file.write(auto)
	file.close()
	#decoded_imgs = autoencoder.predict(x_unlabel)		
	encoder = Model(input_img,encoded)
	encoder.load_weights('auto.hdf5',by_name=True)
	encoder.compile(optimizer='adam', loss='mse')
	en_label = encoder.predict(x_train)
	encoder.save(argv[1]+'1.h5')
	print en_label.shape	
	kmeans = KMeans(n_clusters=10, random_state=0).fit(en_label.reshape((5000,128)))
	la_train = kmeans.labels_
	en_unlabel = encoder.predict(x_unlabel)
	la_unlabel = kmeans.predict(en_unlabel.reshape(45000,128))
	# after cluster
	en_label = np.vstack((en_label,en_unlabel))
	la_train = np.hstack((la_train,la_unlabel))
	
	#CNN
	model = Sequential()
	model.add(Convolution2D(32,3,3,input_shape=(8,4,4),activation='relu',dim_ordering='th',border_mode='same'))	
	model.add(Dropout(0.2))
	model.add(Convolution2D(32, 3, 3,dim_ordering='th',activation='relu',border_mode='same'))
	model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering='th'))
	model.add(Convolution2D(64, 3, 3,dim_ordering='th',activation='relu',border_mode='same'))
	model.add(Dropout(0.2))
	model.add(Convolution2D(64, 3, 3,dim_ordering='th',activation='relu',border_mode='same'))
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
	la_train = np_utils.to_categorical(la_train, 10)
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.fit(en_label,la_train,batch_size=100,nb_epoch=50)
	model.save(argv[1]+'2.h5')
	'''''
	#testing data
	y_test = model.predict_classes(encoder.predict(x_test))
	file = open('test.csv','w')
	file.write('ID,class\n')
	for i in range(10000):
		file.write('{0}'.format(i))
		file.write(',')
		file.write(str(y_test[i]))
		file.write('\n')
	file.close()
	
	print decoded_imgs.shape
	n = 10
	#plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original
		#ax = plt.subplot(2, n, i)
		toimage(x_unlabel[i]).show()
		#ax.get_xaxis().set_visible(False)
		#ax.get_yaxis().set_visible(False)

		# display reconstruction
		#ax = plt.subplot(2, n, i + n)
		toimage(decoded_imgs[i]).show()
		#ax.get_xaxis().set_visible(False)
		#ax.get_yaxis().set_visible(False)
	'''''
if __name__ == '__main__':
	main(sys.argv[1:])