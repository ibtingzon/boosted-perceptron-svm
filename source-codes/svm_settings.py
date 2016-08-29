from svmutil import *
import random

def banana():
	y, x = svm_read_problem('data-sets/banana_data.libsvm')
	prob  = svm_problem(y[:400], x[:400], isKernel=True)

	#Set parameter settings and train
	param = svm_parameter('-s 0 -t 2 -c 72')
	m = svm_train(prob, param)

        #Shuffle data set
	X, Y = [], []
	for i in range(4900):
		randindex = random.randint(0,5299)
		X.append(x[randindex])
		Y.append(y[randindex])
	p_label, p_acc, p_val = svm_predict(Y, X, m)

def splice():
	y, x = svm_read_problem('data-sets/splice_data.libsvm')
	prob  = svm_problem(y[:1000], x[:1000], isKernel=True)

	#Set parameter settings and train
	param = svm_parameter('-s 0 -t 2 -d 1 -c 10')
	m = svm_train(prob, param)

	#Shuffle data set
	X, Y = [], []
	for i in range(2175):
		randindex = random.randint(0,2990)
		X.append(x[randindex])
		Y.append(y[randindex])
	p_label, p_acc, p_val = svm_predict(Y, X, m)
	print p_acc

def main():
	splice()
	#banana()
    
if __name__ == '__main__':
    main()
