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
from keras.models import model_from_json
import pickle

def main(argv):
	label_test =  pickle.load(open(argv[0]+'test.p','rb'))
	label_test = np.array(label_test['data'])
	label_test = label_test.reshape(10000,3,32,32)
	x_test = label_test.astype('float32')/255
	file = open(argv[1]+'.json','r')
	model = model_from_json(file.read())
	model.summary()
	model.load_weights(argv[1]+'.hdf5')
	
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	y_test = model.predict_classes(x_test)
	file = open(argv[2],'w')
	file.write('ID,class\n')
	for i in range(10000):
		file.write('{0}'.format(i))
		file.write(',')
		file.write(str(y_test[i]))
		file.write('\n')
	file.close()


if __name__ == '__main__':
	main(sys.argv[1:])