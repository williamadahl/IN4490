"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import random

        # ninputs = len(inputs[0]) + 1 # one extra for our bias, which is not included in the dataset
        # noutput = len(targets[0])
        # #print(ninputs)
        # #hidden layer weights: Creting a matrix is easiest for keeping track...
        # self.hlw = np.random.uniform(-1,1,(nhidden,ninputs)) # hidden*input+bias, remember to not use
        # self.olw = np.random.uniform(-1,1,(nhidden + 1,noutput)) #hidden+bias*output

def sigmoid(weightedsum):
    #weightedsum = [weightedsum[i] * 2 for i in range(len(weightedsum))]
    #print(weightedsum)
    weightedsum = [1/1+np.exp(-beta * weightedsum[i]) for i in range(len(weightedsum))]
    print(weightedsum)

def matrix(inputs, weights,nhidden):
    weightedsum = []
    for i in range(nhidden):
        answ = 0
        for j in range(len(weights[0])-1):
            answ += weights[i][j]*inputs[j]
        answ += weights[i][ninputs-1] * bias
        weightedsum.append(answ)
    return weightedsum

if __name__ == '__main__':
    inputs = [1,2,3,4]
    bias = 0.5
    beta = 1
    weights = [[1,1,1,1,1],[-2,-2,-2,-2,-2],[3,3,3,3,3]]
    nhidden = 3
    ninputs = 4

    weightedsum = matrix(inputs,weights,nhidden)
    print(weightedsum)
    new = sigmoid(weightedsum)
    print(new)
