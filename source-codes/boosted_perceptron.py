from __future__ import division
import pylab as plt
import numpy as np
import random, math, copy

class Adaboost:
    def __init__(self, K, n):
        self.K = K
        self.alphas = []
        self.learners = []
        self.accuracy = []
        self.K_set = []
        self.H = np.zeros(n)
                
    def adabtrain(self, training_set, test_set):
        N = len(training_set)
        w = np.ones(N)/N #weights
        d = 2   #dimension
        S = training_set
        indexes = np.arange(N)
        
        for t in range(self.K + 1):
            #Select new training set St from S according to prob dist. w
            indexes = np.arange(N)
            indexes = np.random.choice(indexes, size=N, replace=True, p=w)
            St = [S[indexes[i]] for i in range(N)]

            #Train L on St and get hypothesis h            
            #eps = 0.5
            #while (eps >= 0.5): 
            learner = Perceptron(d)
            learner.classify(St)
            h = np.array(learner.predict(S)) 
            y = np.array([S[i][1] for i in range(N)])
            eps = sum(w[h != y]) #Compute Training Error (eps)

            #Compute alpha and update weights
            alpha = float(0.5)*np.log((1-eps)/eps)
            w = np.array([w[i]*(np.exp(-alpha*y[i]*h[i])) for i in range(N)])
            w /= w.sum()

            self.alphas.append(alpha)
            self.learners.append(learner)

            #AdabPredict - produce result at increments of 10
            rnge = [i*10 for i in range(1,100)]
            if (t in rnge):
                self.adabpredict(t, test_set)
        self.adabplot()

    def adabpredict(self, K, test_set):
        N = len(test_set)
        H, y = np.zeros(N), []
        accuracy, count = 0, np.ones(N)
        y = [test_set[i][1] for i in range(N)]
        inc = 10

        #Predict with Adaboost 
        for t in range(K-inc,K): #results at increments of 10
            h = self.learners[t].predict(test_set)
            h = np.array(h).astype(float)
            self.H = self.H + (self.alphas[t]*h)
        H = np.sign(self.H)

        #Calculate Accuracy
        count = sum(count[y == H])
        accuracy = (float(count)/N)*100
        print "Adaboost test accuracy ", accuracy, "% (", count, "/", N, ") K = ", K 
        
        self.accuracy.append(accuracy)
        self.K_set.append(K)

    def adabplot(self):
        plt.plot(self.K_set, self.accuracy)
        plt.xlabel("K (rounds)")
        plt.ylabel("Accuracy")
        plt.show()
            
class Perceptron:
    def __init__(self, d):
        self.d = d
        self.w = []
        self.error = 0
        self.accuracy = 0

    def generate_data(self, N):
        mu1, mu2, sigma = 0, 10, 1
        training_set = []
        test_set = []
        for k in range(0,N):
            x1 = ([np.random.normal(mu1, sigma) for i in range(self.d)])
            x2 = ([np.random.normal(mu2, sigma) for i in range(self.d)])
            if (k < N/2):
                training_set.append([x1,-1])
                training_set.append([x2,1])
            else:
                test_set.append([x1,-1])
                test_set.append([x2,1])
        np.random.shuffle(training_set)
        np.random.shuffle(test_set)
        return training_set, test_set

    def classify(self, training_set):
        N = len(training_set)
        itercnt, maxitercnt = 0, 10000
        nv, nw, bias = float(0), float(0), float(1)
        v, w = np.zeros(self.d + 1), np.zeros(self.d + 1)
        ht = []
        
        while (itercnt < maxitercnt):
            randsamp = training_set[random.randint(0,N-1)] #random sample
            x, y = copy.deepcopy(randsamp[0]), copy.deepcopy(randsamp[1])
            x.insert(0, bias) 
            x = np.array(x)
            y_hat = 1 if (np.dot(v,x) >= 0) else -1
            if (y_hat*y > 0):
                nv = nv + 1
            else:
                if (nv > nw): 
                    w = v
                    nw = nv
                v = v + y*x
                nv = 0
            itercnt = itercnt + 1
        self.w = w
        
    def predict(self, test_set):
        N, bias, ht = len(test_set), float(1), []
        accuracy = 0

        for i in range(N):
            x, y = copy.deepcopy(test_set[i][0]), copy.deepcopy(test_set[i][1])
            x.insert(0, bias)
            x = np.array(x)
            y_hat = 1 if (np.dot(self.w,x) >= 0) else -1
            ht.append(y_hat)

            if(y*y_hat > 0): 
                self.accuracy += 1
            else:
                self.error += 1
        
        self.error = (float(self.error)/N)*100
        self.accuracy = (float(self.accuracy)/N)*100

        return ht

def main():
    #filename = "data-sets/banana_data.csv"
    filename = "data-sets/splice_data.csv"
    get_data = np.genfromtxt(filename, delimiter=',')
    training_set, test_set = [],[]
    dim = 2
    K = 30

    data = np.array(get_data)
    np.random.shuffle(data)
    
    for i in range(len(data)):
        x, y = [], data[i][0]
        for j in range(dim):
            x.append(data[i][j+1])
        #if(i < 400): #banana_data 
        if(i < 1000): #splice_data
            training_set.append([x,y])

    data = np.array(get_data)
    np.random.shuffle(data)

    for i in range(len(data)):
        x, y = [], data[i][0]
        for j in range(dim):
            x.append(data[i][j+1])
        #if(i < 4900): #banana_data 
        if(i < 2175): #splice_data
            test_set.append([x,y])

    np.random.shuffle(training_set)
    np.random.shuffle(test_set)

    p = Perceptron(dim)
    #training_set, test_set = p.generate_data(100)
    p.classify(training_set)
    p.predict(training_set)
    p.predict(test_set)

    test = test_set
    #test = training_set

    a = Adaboost(K, len(test))
    a.adabtrain(training_set, test)
    
if __name__ == '__main__':
    main()
