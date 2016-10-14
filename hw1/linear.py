import numpy as np 
import math
import random
def compute_loss_function(b,w,regulation_delta,final_data):
	total_loss = 0
	for i in range(12):
		for a in range(471):
			x = final_data[a+i*480:a+9+i*480,0:10]
			w_multiply_data  = np.multiply(w,x)
			w_multiply_data = np.sum(w_multiply_data)
			total_loss += (final_data[9+a+i*480,9] - (b + w_multiply_data))**2 
	total_loss+=np.sum(np.square(w))*regulation_delta
	return total_loss		
def gradient_descent(b_starting,w_starting,learning_rate,regulation_delta,num_iteration,final_data):
	b = b_starting
	w = w_starting
	acceleration_b = 0
	acceleration_w = 0
	prev_loss =0
	loss = 41651
	for u in range(num_iteration):		
		for i in range(12):
			differential_b = 0
			differential_w = np.zeros((9,10))
			for a in range(471):
				#a = random.randint(0,470)
				x = final_data[a+i*480:a+9+i*480,0:10]
				w_multiply_data  = np.multiply(w,x)
				w_multiply_data = np.sum(w_multiply_data)
				diff_b = -2*(final_data[9+a+i*480,9] - (b + w_multiply_data)) 
				diff_w = -2*x*(final_data[9+a+i*480,9] - (b + w_multiply_data)) + 2*w*regulation_delta
				differential_b += diff_b
				differential_w += diff_w
				acceleration_b += diff_b**2
				acceleration_w += np.square(diff_w)
				gradient_b = differential_b
				gradient_w = differential_w 
				b =  b - (learning_rate*gradient_b/math.sqrt(0.00000000001+ acceleration_b)) 
				w = w - (learning_rate*gradient_w/np.sqrt(0.00000000001+acceleration_w))
				differential_b = 0
				differential_w = np.zeros((9,10))
		prev_loss = loss
		loss = compute_loss_function(b,w,regulation_delta,final_data)
		if abs(prev_loss - loss) < 0.15:
			break
		#print loss	
	return b,w 

def main():
	#read data
	input_buffer = []
	arr = []
	with open('train.csv') as textFile:
		next(textFile)
		for line in textFile:
			input_buffer = line.strip().split(',')
			for i in range (3):
				input_buffer.pop(0)
			arr.append(input_buffer)
	#print arr
	# data all into an np.array
	np_data = np.array(arr)
	#print np_data.shape
	np_data[np_data == 'NR'] = '0'

	#print np_data
	final_data = []
	for i in range(240):
		final_data.append(np_data[18*i:18*(i+1),:].T)
	final_data = np.array(final_data)
	final_data = final_data.astype(np.float)
	final_data = final_data.reshape(240*24,18)
	#print final_data.shape
	#parameters
	learning_rate = 0.01
	regulation_delta = 0
	num_iteration = 5000
	#weight and bias
	w_initial = np.zeros((9,10))
	b_initial = 0
	#computing the loss function
	initial_loss = compute_loss_function(b_initial,w_initial,regulation_delta,final_data)
	#print initial_loss
	#start learning by loss function gradient descent
	new_b,new_w =gradient_descent(b_initial,w_initial,learning_rate,regulation_delta,num_iteration,final_data)
	#after learning
	final_loss = compute_loss_function(new_b,new_w,regulation_delta,final_data)
	#print new_w
	#print final_loss
	#test data
	test = []
	with open('test_X.csv') as textFile:
		for line in textFile:
			input_buffer = line.strip().split(',')
			for i in range (2):
				input_buffer.pop(0)
			test.append(input_buffer)
	np_test = np.array(test)
	np_test[np_test == 'NR'] = '0'
	test_data = []
	for i in range(240):
		test_data.append(np_test[18*i:18*(i+1),:].T)
	test_data = np.array(test_data)
	test_data = test_data.astype(np.float)
	test_data = test_data.reshape(240*9,18)
	file = open('linear_regression.csv','w')
	file.write('id,value')
	file.write('\n')
	for i in range (240):
		x = test_data[9*i:9*(i+1),0:10]
		predict_y = np.sum(np.multiply(x,new_w)) + new_b
		file.write('id_{0}'.format(i))
		file.write(',')
		file.write(str(predict_y))
		file.write('\n')
		
	file.close()
if __name__ == '__main__':
	main()