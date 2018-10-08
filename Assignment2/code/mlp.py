"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import random

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.bias = 1
        self.eta = 0.1
        self.momentum = 0.0
        ninputs = len(inputs[0]) + 1 # one extra for our bias, which is not included in the dataset
        noutput = len(targets[0])
        #print(ninputs)
        #hidden layer weights: Creting a matrix is easiest for keeping track...
        self.hlw = np.random.uniform(-1,1,(nhidden,ninputs)) # hidden*input+bias, remember to not use
        self.olw = np.random.uniform(-1,1,(nhidden + 1,noutput)) #hidden+bias*output


        print('To be implemented')

    # You should add your own methods as well!

    def earlystopping(self, inputs, targets, valid, validtargets):
        print('first')
        print('inputs\n',inputs[0]) # input matrices list in list
        print('targets\n',targets[0]) # input matrices list in list
        print('valid\n', valid[0]) # validation set
        print('validtargets\n', validtargets[0]) # validation targets for our test



        print('To be implemented')

    def train(self, inputs, targets, iterations=100):
        print('To be implemented')

    def forward(self, inputs):
        print(self)

        print('To be implemented')

    def confusion(self, inputs, targets):
        print('To be implemented')

    # helper function for sigmoid on each neuron
    def sigmoid(weightedsum):
        weightedsum = [1/1+np.exp(-beta *weightedsum[i]) for i in range(len(weightedsum))]
        return weightedsum

    # helper funcion for calulating weightetsum for each neuron
    def matrix(inputs, weights):
        weightedsum = []
        for i in range(nhidden):
                answ = 0
                for j in range(len(weights[0])-1):
                    answ += weights[i][j]*inputs[j]
                answ += weights[i][ninputs-1] * self.bias
                weightedsum.append(answ)

        return weightedsum
