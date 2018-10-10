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

def matrix(inputs, weights,nhidden, number):
    weightedsum = []
    print(inputs[number][1])
    for x in range(2):
        for i in range(nhidden):
            answ = 0
            for j in range(len(weights[0])-1):
                answ += weights[i][j]*inputs[x][j]
                answ += weights[i][ninputs-1] * bias
                weightedsum.append(answ)
    return weightedsum


def error_test(delta_hidden, weights, errors):
    ret = []
    for i in range(len(hidden_errors)):
        answ = 0
        for j in range(len(errors)):
            answ += weights[j][i] * errors[j]

        ret.append(answ)

    return ret

def update_weights_outer(weights2,eta,hidden_activate,errors):

    print(weights2)
    print('weights', np.shape(weights2))
    print('errors', np.shape(errors))
    print('hidden activate', np.shape(hidden_activate))
    #print(len(weights2))
    for i in range(len(weights2)):

    #    print(len(weights2[0]))
        for j in range(len(hidden_activate)):
        #    print(weights2[i])
            #print('error',errors[i])
            weights2[i][j] = (weights2[i][j] - (hidden_activate[j]*errors[i]*eta))

        #    print(answ)

    print(weights2)

def update_inner(weights2, eta, inn, error):
    print('inner')
    print('weights: ',np.shape(weights2))
    print('error: ',np.shape(error))
    print('inn: ',np.shape(inn))

    


    for i in range(len(weights2)):
        for j in range(len(inn)):
        #    print(weights2[i])
            weights2[i][j] = (weights2[i][j]-(eta*error[i]*inn[j]))

    print(weights2)

if __name__ == '__main__':
    # inputs = [[1,2,3,4],[4,3,2,1]]
    # bias = 0.5
    # beta = 1
    # weights = [[1,1,1,1,1],[-2,-2,-2,-2,-2],[3,3,3,3,3]]
    # errors = [2,3]
    # delta_hidden =[0,0,0]
    # nhidden = 3
    # ninputs = 4
    # number = 0
#    delta_hidden = error_test(delta_hidden, weights2,errors)


    weights2 = [[1,2,3],[2,2,2]]
    eta = 0.1
    inn = [1,2,1]
    error = [0.5, 1]
    #new_weights = update_weights_outer(weights2, eta, hidden_activate,errors)
    update_inner(weights2, eta, inn, error)
#weightedsum = matrix(inputs,weights,nhidden,number)
    #print(weightedsum)
    #new = sigmoid(weightedsum)
    #print(new)
