import numpy as np 
import math
import random
import sys
def sigmoid(x) :
	return 1 / (1 + np.exp(-x))
def compute_loss_function(b1,b2,w1,w2,final_data,y):
	total_loss = 0
	w1_multiply_data  = np.dot(final_data,w1) + b1
	predict_y1 = sigmoid(w1_multiply_data)
	w2_multiply_data  = np.dot(predict_y1,w2) + b2
	predict_y2 = sigmoid(w2_multiply_data)
	total_loss = np.sum(-((y * np.log(predict_y2+0.0001) ) + ((1- y) * np.log(1 - predict_y2 + 0.0001)  )) )
	#for i in range(4001):
		#total_loss += -(final_data[i,0]*math.log(predict_y2[i,0]+0.0001) + (1- final_data[i,0]) * math.log(1 - predict_y2[i,0] + 0.0001) )
	return total_loss		
def gradient_descent(b1_starting,b2_starting,w1_starting,w2_starting,learning_rate,num_iteration,final_data,y):
	b1 = b1_starting
	w1 = w1_starting
	b2 = b2_starting
	w2 = w2_starting

	acceleration_b1 = 0
	acceleration_w1 = np.zeros((64,5))
	acceleration_b2 = 0
	acceleration_w2 = np.zeros((5,1))

	for u in range(num_iteration):
		differential_b1 = 0
		differential_w1= np.zeros((64,5))
		differential_b2 = 0
		differential_w2= np.zeros((5,1))

		w1_multiply_data  = np.dot(final_data,w1) + b1
		predict_y1 = sigmoid(w1_multiply_data)
		#Sprint predict_y1.shape
		w2_multiply_data  = np.dot(predict_y1,w2) + b2
		predict_y2 = sigmoid(w2_multiply_data)
		#print predict_y2.shape
		differential_b2 = (-(y - predict_y2))
		differential_w2 = (-np.dot(predict_y1.T, -differential_b2 ) )
		differential_b1 =  -np.dot(-differential_b2,w2.T)*(predict_y1 - np.square(predict_y1) )
		differential_w1 =  -np.dot(final_data.T,-differential_b1 )
		#print differential_w1
		#print differential_w2
		acceleration_b2 += np.square(np.sum(differential_b2))
		acceleration_w2 += np.square(differential_w2)


		acceleration_b1 += np.square(np.sum(differential_b1))
		acceleration_w1 += np.square(differential_w1)
		#print acceleration_w1.shape
		#print acceleration_w2.shape
		differential_b1 = np.sum(differential_b1)
		differential_b2 = np.sum(differential_b2)
		b2 =  b2 - (learning_rate*differential_b2/math.sqrt(0.00000000001+ acceleration_b2)) 
		w2 = w2 - (learning_rate*differential_w2/np.sqrt(0.00000000001+acceleration_w2))
		b1 =  b1 - (learning_rate*differential_b1/math.sqrt(0.00000000001+ acceleration_b1)) 
		w1 = w1 - (learning_rate*differential_w1/np.sqrt(0.00000000001+acceleration_w1))
		loss = compute_loss_function(b1,b2,w1,w2,final_data,y)

		print loss
	return b1,w1,b2,w2 

def main(argv):
	#read data
	input_buffer = []
	arr = []
	y = []
	with open(argv[0]) as textFile:
		for line in textFile:
			input_buffer = line.strip().split(',')
			for i in range (1):
				input_buffer.pop(0)
			input_buffer = [float(i) for i in input_buffer]
			buf = input_buffer[57]
			del input_buffer[57]
			input_buffer.append((input_buffer[50])**0.5)
			input_buffer.append((input_buffer[51])**0.5)
			input_buffer.append((input_buffer[52])**0.5)
			input_buffer.append((input_buffer[53])**0.5)
			input_buffer.append((input_buffer[54])**0.5)
			input_buffer.append((input_buffer[55])**0.5)
			input_buffer.append((input_buffer[56])**0.5)
			y.append(buf)
			arr.append(input_buffer)
			
	#data all into np array
	np_data = np.array(arr)
	np_y = np.array(y)
	np_y = np_y.T
	np_y = np_y.reshape(4001,1)
	#np_data = np_data.astype(np.float)
	#parameters
	learning_rate = 0.1
	num_iteration = 50000
	#weight and bias
	w1_initial = np.random.uniform(-0.1,0.1,(64,5))
	w2_initial = np.random.uniform(-0.1,0.1,(5,1))
	#w_initial = np.random.uniform(-0.1,0.1,(64,1))
	b1_initial = 0.01
	b2_initial = -0.02
	#computing the loss function
	initial_loss = compute_loss_function(b1_initial,b2_initial,w1_initial,w2_initial,np_data,np_y)
	print initial_loss
	#start learning by loss function gradient descent
	new_b1,new_w1,new_b2,new_w2 =gradient_descent(b1_initial,b2_initial,w1_initial,w2_initial,learning_rate,num_iteration,np_data,np_y)
	#test data
	

	file = open(argv[1],'w')
	for i in range(64):
		for j in range(5):
			x = new_w1[i,j]
			file.write(str(x))
			if j != 4:
				file.write(',')
		file.write('\n')
	for i in range(5):
		x = new_w2[i,0]
		file.write(str(x))
		file.write('\n')	
	file.write(str(new_b1))
	file.write('\n')
	file.write(str(new_b2))
	file.close()
if __name__ == '__main__':
	main(sys.argv[1:])