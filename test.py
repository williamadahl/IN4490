"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import random
from collections import deque

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
# new_error = np.sum((o - validtargets)**2) / n
def sum_squares(real, target):
    #print(real)
    #print(target)
    error = np.sum((np.array(real)-np.array(target))**2)/len(target)
    #print(error )
    print(np.argmax(real))
    print(np.argmax(target))

    # for i in range(2):
    #     test = []
    #     for j in range(3):
    #         numb = np.random.rand(1,3)
    #         print(numb)
    #         test.append(numb)
    #     print('what',test)


def shift(sequence, shift):
    return sequence[-shift:] + sequence[:-shift]


def datasplit(data, targets, folds):

    # print('Befor shift: ', data)
    # data = shift(data,1)
    # print('After shift : ', data)
    #

    foldsize = len(data)//folds
    test_start_index = 0
    test_end_index = foldsize

    valid_start_index = foldsize
    valid_stop_index = foldsize*2

    training_start_index = valid_stop_index + foldsize

    for i in range(len(data)):
        print('Fold number : ',i)
        test_data = data[0:foldsize]
        valid_data = data[valid_start_index:valid_stop_index]
        training_data = data[training_start_index:]
        print('TEST: ', test_data)
        print('VALID: ', valid_data)
        print('TRAINING', training_data)
        print('THIS IS DATA : ', data)
        data = shift(data,3)
        print('THIS IS DATA AFTER: ', data)


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
    real = [1,2,3]
    target = [1,1,1]
    #sum_squares(real,target)

    weights2 = [[1,2,3],[2,2,2]]
    eta = 0.1
    inn = [1,2,1]
    error = [0.5, 1]
    folds = 4
    data = [[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]]
    targets = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1]]
    datasplit(data, targets, folds)
    #new_weights = update_weights_outer(weights2, eta, hidden_activate,errors)
    #update_inner(weights2, eta, inn, error)
    #weightedsum = matrix(inputs,weights,nhidden,number)
    #print(weightedsum)
    #new = sigmoid(weightedsum)
    #print(new)
