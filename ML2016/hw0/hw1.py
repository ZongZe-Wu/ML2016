from numpy import loadtxt,savetxt,zeros
import sys
def main(argv):
	with open(argv[1]) as textFile:
		arr = [line.split() for line in textFile]
	a = int(argv[0])
	array = []
	for i in range(len(arr)):
		array.append(float(arr[i][a]))
	array.sort()
	file = open('ans1.txt','w')
	for i in range(len(array) - 1):
		file.write('{0}'.format(array[i]))
		file.write(',')
	file.write('{0}'.format(array[len(array)-1]))
	file.close()
if __name__ == '__main__':
	main(sys.argv[1:])