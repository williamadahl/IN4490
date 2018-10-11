"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import random
import sys
from sklearn.metrics import confusion_matrix

class mlp:
    def __init__(self, inputs, targets, nhidden):
    #    nhidden = 3
        self.beta = 1
        self.bias = 1
        self.eta = 0.1 # learning rate
        self.momentum = 0.0
        ninputs = len(inputs[0]) + 1 # one extra for our bias, which is not included in the dataset
        noutput = len(targets[0])
        self.output_nr = noutput
        self.nhidden = nhidden
        self.ninputs = ninputs

        self.hlw = np.random.uniform(-1,1,(nhidden,ninputs)) # hidden*input+bias, remember to not use
        self.olw = np.random.uniform(-1,1,(noutput, nhidden + 1)) #hidden+bias*output



        print('To be implemented')

    # You should add your own methods as well!

    def earlystopping(self, inputs, targets, valid, validtargets):

        while 1:
            #average_error_list = []
            self.train(inputs, targets, 1) # actuall training
        #    error_list = [] # list of all errors, fresh for each iteration of training
            #iterate over trainingset to create an average error

            correct_guesses = 0
            for i in range(len(valid)):
                hidden , output = self.forward(valid[i]) # the testing
                index_validtargets = np.argmax(validtargets[i])
                index_output = np.argmax(output)
            #    print('Valid target : ', validtargets[i])
                #print('ouptut : ', output)
                if(index_validtargets == index_output):
                    correct_guesses += 1

            #average_error = np.mean(error_list)
            #average_error_list.append(average_error)
            print('number of correct guesses: ', correct_guesses)

        #    print(average_error_list)
            # of we get a prediction of about 90% we can be happy and end the training
            # can implement exit if x number of iterations does not yeild improvement
            if correct_guesses >= 80:
                break


    def train(self, inputs, targets, iterations):
        for it in range(iterations):
            for i in range(len(inputs)):
                #forward
                hidden_out, final_output = self.forward(inputs[i])
                #backward
                self.back_prop(hidden_out,final_output,targets[i],inputs[i])


# TODO: continue from this line



    def forward(self, vector):
        hidden_out = self.matrix_hidden(vector)
        hidden_out = self.sigmoid(hidden_out)
        final_output = self.matrix_outer(hidden_out)
        final_output = self.activation_outer(final_output)
        return hidden_out, final_output

    def back_prop(self, hidden_output, final_output,targets,vector):
        eo = self.error_outer(final_output, targets) # delta outer
        ei = self.error_hidden(hidden_output,eo,targets) # delta inner
        #ready to update values for olw.
        self.update_weights_outer(hidden_output, eo)
        self.update_weights_inner(vector,ei)

    def update_weights_inner(self,vector,ei):
        for i in range(len(self.hlw)):
            for j in range(len(vector)):
                self.hlw[i][j] = self.hlw[i][j] - (self.eta *ei[i] * vector[j])

        for i in range(len(self.hlw)):
            self.hlw[i][len(self.hlw[0])-1] = self.hlw[i][len(self.hlw[0])-1] - (self.eta*ei[i]*self.bias)

    # hidden_output is the activation from hidden
    def update_weights_outer(self, hidden_output, eo):
        for i in range(len(self.olw)):
            for j in range(len(hidden_output)):
                self.olw[i][j] = (self.olw[i][j]-hidden_output[j]*eo[i]*self.eta)

        for i in range(len(self.olw)):
            self.olw[i][len(self.olw[0])-1] = self.olw[i][len(self.olw[0])-1] - (self.eta*eo[i]*self.bias)


    # 4.9 from the book
    # calculate the delta for each neuron in hidden layer
    def error_hidden(self, hidden_output, eo,targets):
        ei = []
        for i in range(len(hidden_output)):
            answ = 0
            # loop over all indexes downwards in the matrix
            for j in range(len(targets)):
                answ += self.olw[j][i]*eo[j]
            ei.append(answ)
        return ei

    # 4.14 from the book
    def error_outer(self, final_output, targets):
        eo = [(final_output[i]-targets[i]) for i in range(len(final_output))] # these are the deltas
        return(eo)


    def confusion(self, inputs, targets):

        actual = [0]*len(inputs)
        predicted = [0]*len(inputs)

        for i in range(len(inputs)):
            hidden , output = self.forward(inputs[i]) # the testing
            index_out = np.argmax(output)
            predicted[i] += index_out

        for i in range(len(targets)):
            index = np.argmax(targets[i])
            actual[i] += index

        #print(actual)
        #print(predicted)
        result = confusion_matrix(actual, predicted)
        print(result)


    # this is a linear function, might change this later, but ill let it stand for now.
    def activation_outer(self, weightedsum):
        return weightedsum

    # helper function for sigmoid on each neuron
    def sigmoid(self, weightedsum):
        weightedsum = [1/(1+np.exp(-self.beta *weightedsum[i])) for i in range(len(weightedsum))]
        return weightedsum

    # helper funcion for calulating weightetsum for each neuron
    def matrix_hidden(self, vector):
        weightedsum = []
        for i in range(self.nhidden):
            answ = 0
            for j in range(len(self.hlw[0])-1):
                answ += self.hlw[i][j]*vector[j]
            answ += self.hlw[i][self.ninputs-1] * self.bias
            weightedsum.append(answ)
        #returns an array of the weighted sums.
        return weightedsum

    def matrix_outer(self,hidden_out):
        weightedsum = []
        for i in range(self.output_nr):
            answ = 0
            for j in range(len(self.olw[0])-1):
                answ += self.olw[i][j]*hidden_out[j]
            answ += self.olw[i][(len(hidden_out))-1] * self.bias
            weightedsum.append(answ)
        #returns an array of the weighted sums.
        return weightedsum
