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
from keras.models import load_model
import pickle

def main(argv):
	label_test =  pickle.load(open(argv[0]+'test.p','rb'))
	label_test = np.array(label_test['data'])
	label_test = label_test.reshape(10000,3,32,32)
	x_test = label_test.astype('float32')/255
	model1 = load_model(argv[1]+'1.h5')
	model2 = load_model(argv[1]+'2.h5')	
	model1.compile(optimizer='adam', loss='mse')
	model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	y_test = model2.predict_classes(model1.predict(x_test))
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