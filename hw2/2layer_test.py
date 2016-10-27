import numpy as np 
import math
import random
import sys
def sigmoid(x) :
	return 1 / (1 + np.exp(-x))
def main(argv):
	input_buffer = []
	w1_buff = []
	b1_buff = 0
	w2_buff = []
	b2_buff = 0
	i = 1
	with open(argv[0]) as textFile:
		for line in textFile:
			if i <= 64 :
				input_buffer = line.strip().split(',')
				w1_buff.append(input_buffer)
			if i >= 65 and i <=69:
				input_buffer = line.strip()
				w2_buff.append(input_buffer)
			if i == 70:
				b1_buff = float(line.strip())
			if i == 71:
				b2_buff = float(line.strip())
			i += 1
	new_w1 = np.array(w1_buff)
	new_w1 = new_w1.astype(np.float)
	new_w2 = np.array(w2_buff)
	new_w2 = new_w2.astype(np.float)
	new_b1 = b1_buff	
	new_b2 = b2_buff
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
	file.write('id,label')
	file.write('\n')
	predict_y1 = (np.dot(np_test,new_w1) ) + new_b1
	predict_y1 = sigmoid(predict_y1)
	predict_y2 = (np.dot(predict_y1,new_w2) ) + new_b2
	predict_y2 = sigmoid(predict_y2)
	for i in range(600):
		file.write('{0}'.format(i+1))
		file.write(',')
		if predict_y2[i] >= 0.5:
			file.write("1")
			file.write('\n')
		else:
			file.write("0")
			file.write('\n')

	file.close()

if __name__ == '__main__':
	main(sys.argv[1:])