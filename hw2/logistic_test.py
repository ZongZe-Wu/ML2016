import numpy as np 
import math
import random
import sys
def sigmoid(x) :
	return 1 / (1 + np.exp(-x))
def main(argv):
	input_buffer = []
	w_buff = []
	b_buff = 0
	with open(argv[0]) as textFile:
		for line in textFile:
			input_buffer = line.strip().split()
			w_buff.append(input_buffer)
			b_buff = float(line.strip())
		del w_buff[64]
	new_w = np.array(w_buff)
	new_w = new_w.astype(np.float)
	new_b = b_buff		
	test = []
	with open(argv[1]) as textFile:
		for line in textFile:
			input_buffer = line.strip().split(',')
			for i in range (1):
				input_buffer.pop(0)
			input_buffer = [float(i) for i in input_buffer]
			input_buffer.append((input_buffer[50])**0.5)
			input_buffer.append((input_buffer[51])**0.5)
			input_buffer.append((input_buffer[52])**0.5)
			input_buffer.append((input_buffer[53])**0.5)
			input_buffer.append((input_buffer[54])**0.5)
			input_buffer.append((input_buffer[55])**0.5)
			input_buffer.append((input_buffer[56])**0.5)
			test.append(input_buffer)
	np_test = np.array(test)
	#np_test = np_test.astype(np.float)
	file = open(argv[2],'w')
	#y_file = open('predict_y.csv','w')
	file.write('id,label')
	file.write('\n')
	predict_y = (np.dot(np_test,new_w) ) + new_b
	predict_y = sigmoid(predict_y)
	for i in range(600):
		file.write('{0}'.format(i+1))
		file.write(',')
		if predict_y[i] >= 0.5:
			file.write("1")
			file.write('\n')
		else:
			file.write("0")
			file.write('\n')
		#y_file.write(str(predict_y))
		#y_file.write('\n')
	file.close()
	#y_file.close()



if __name__ == '__main__':
	main(sys.argv[1:])