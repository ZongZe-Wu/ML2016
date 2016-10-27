import numpy as np 
import math
import random
import sys
def sigmoid(x) :
	return 1 / (1 + np.exp(-x))
def compute_loss_function(b,w,final_data,y):
	total_loss = 0
	w_multiply_data  = np.dot(final_data,w) + b
	predict_y = sigmoid(w_multiply_data)
	total_loss = (-(np.dot(y , np.log(predict_y+0.0001)) + np.dot((1- y) , np.log(1 - predict_y + 0.0001) ) ) )
	#total_loss1 = np.sum(-(y * np.log(predict_y+0.0001)) + ((1- y) * np.log(1 - predict_y + 0.0001)  ) )
	#for i in range(4001):
		#total_loss += -(final_data[i,0]*math.log(predict_y[i,0]+0.0001) + (1- final_data[i,0]) * math.log(1 - predict_y[i,0] + 0.0001) )


	return total_loss		
def gradient_descent(b_starting,w_starting,learning_rate,num_iteration,final_data,y):
	b = b_starting
	w = w_starting
	acceleration_b = 0
	acceleration_w = np.zeros(64)
	for u in range(num_iteration):
		differential_b = 0
		differential_w = np.zeros((64,1))
		w_multiply_data  = np.dot(final_data,w) + b
		predict_y = sigmoid(w_multiply_data)
		differential_b = (-(y - predict_y))
		differential_w = (-np.dot(final_data.T, (y- predict_y) ) )
		acceleration_b += np.square(np.sum(differential_b))
		acceleration_w += np.square(differential_w)
		differential_b = np.sum(differential_b)
		b =  b - (learning_rate*differential_b/math.sqrt(0.00000000001+ acceleration_b)) 
		w = w - (learning_rate*differential_w/np.sqrt(0.00000000001+acceleration_w))
		loss = compute_loss_function(b,w,final_data,y)
		print loss
	return b,w 

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

	#np_data = np_data.astype(np.float)
	#parameters
	learning_rate = 0.1
	num_iteration = 50000
	#weight and bias
	w_initial = np.zeros(64)
	#w_initial = np.random.uniform(-0.1,0.1,(64,1))
	b_initial = 0
	#computing the loss function
	initial_loss = compute_loss_function(b_initial,w_initial,np_data,np_y)
	print initial_loss
	#start learning by loss function gradient descent
	new_b,new_w =gradient_descent(b_initial,w_initial,learning_rate,num_iteration,np_data,np_y)
	#test data
	

	file = open(argv[1],'w')
	for i in range(64):
		x = new_w[i]
		file.write(str(x))
		file.write('\n')
	file.write(str(new_b))
	file.close()
if __name__ == '__main__':
	main(sys.argv[1:])