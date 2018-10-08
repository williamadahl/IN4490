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
        self.nhidden = nhidden
        self.ninputs = ninputs

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

        # can do a test first and check vs error, if worse go train more. while true
        #hidden_output = train(self.hlw, inputs)
        #print(hidden_output)
        hidden_output, final_output = self.forward(inputs)
        print(hidden_output)
        print(final_output)


        print('To be implemented')

    def train(self, inputs, targets, iterations=100):
        print('To be implemented')

    def forward(self, inputs):
        # inner layer output
        print(inputs[0])
        print('this ex')
        print(self.hlw[0])

        hidden_out = self.matrix(inputs)
        print(hidden_out)
        # send to sigmoid
        # outer layer output
        final_output = self.matrix(hidden_out)
        print(final_output)
        # send to sigmoid
        return hidden_out, final_output

    def confusion(self, inputs, targets):
        print('To be implemented')

    # helper function for sigmoid on each neuron
    def sigmoid(weightedsum):
        weightedsum = [1/1+np.exp(-beta *weightedsum[i]) for i in range(len(weightedsum))]
        return weightedsum

    # helper funcion for calulating weightetsum for each neuron
    def matrix(self, inputs):
        weightedsum = []
        print('here is nhidden', self.nhidden)
        print('here is hlw[0][0]',self.hlw[0][0])

        for i in range(self.nhidden):
            print('i',i)
            answ = 0
            print(len(self.hlw[0]))
            print(len(self.hlw[1]))
            print(len(self.hlw[11]))
            print(inputs[0])
            for j in range(len(self.hlw[0])-1):
                print('j',j)
                answ += self.hlw[i][j]*inputs[j]
            answ += self.hlw[i][self.ninputs-1] * self.bias
            weightedsum.append(answ)

        return weightedsum
