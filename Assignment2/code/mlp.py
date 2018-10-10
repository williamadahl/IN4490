"""
    This pre-code is a nice starting point, but you can
    change it to fit your needs.
"""
import numpy as np
import random
import sys

class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.bias = 1
        self.eta = 0.1 # learning rate
        self.momentum = 0.0
        ninputs = len(inputs[0]) + 1 # one extra for our bias, which is not included in the dataset
        noutput = len(targets[0])
        self.output_nr = noutput
        self.nhidden = nhidden
        self.ninputs = ninputs

        #print(ninputs)
        #hidden layer weights: Creting a matrix is easiest for keeping track...
        self.hlw = np.random.uniform(-1,1,(nhidden,ninputs)) # hidden*input+bias, remember to not use
        # self.olw = np.random.uniform(-1,1,(nhidden + 1,noutput)) #hidden+bias*output
        self.olw = np.random.uniform(-1,1,(noutput, nhidden + 1)) #hidden+bias*output



        print('To be implemented')

    # You should add your own methods as well!

    def earlystopping(self, inputs, targets, valid, validtargets):
        #remember to use 4.13 for output activation function
        print('first')
        print('inputs\n',inputs[0]) # input matrices list in list
        print('targets\n',targets[0]) # input matrices list in list
        print('valid\n', valid[0]) # validation set
        print('validtargets\n', validtargets[0]) # validation targets for our test
        print('lenght of inputs : ', len(inputs))

        # can do a test first and check vs error, if worse go train more. while true
        #hidden_output = train(self.hlw, inputs)
        #print(hidden_output)
    #    final_hidden_output, final_output = self.forward(inputs)
        test = self.train(inputs, targets,100)
        print(final_hidden_output)
        print(final_output)


        print('To be implemented')

    def train(self, inputs, targets, iterations):
        # need a counter to specify which input array we are working on
        print('To be implemented')
        print('number of iterations:',iterations)
        for it in range(iterations):
            for i in range(len(inputs)):
                #forward
                hidden_out, final_output = self.forward(inputs[i])
                print('In train hidden ', hidden_out)
                print('In train output ', final_output)
                #backward
                self.back_prop(hidden_out,final_output,targets[i],inputs[i])

# TODO: continue from this line



    def forward(self, vector):
        print(vector)
        # inner layer output
        #print('FORWARD inputs: ', len(inputs))
        #print(inputs[0])
        ##print('this ex')
        #print(self.hlw[0])
    #    print('before: ', self.hlw[0])
        # create hidden_output from one training sample
        hidden_out = self.matrix_hidden(vector)
        print('First round hidden:', hidden_out)
        hidden_out = self.sigmoid(hidden_out)
        print('After sigmoid hidden', hidden_out)
        final_output = self.matrix_outer(hidden_out)
        print('First round outer: ', final_output)
        final_output = self.activation_outer(final_output)
        #    print('After activation outer:', final_output)
            # now i will call back propagation on my Values to update the weight matrices
        #    print('after: ', self.hlw[0])


    #    print(hidden_out)
    #    print('DONE')
        # send to sigmoid
        # outer layer output
#        final_output = self.matrix(hidden_out)
    #    print(final_output)
        # send to sigmoid
        return hidden_out, final_output

    def back_prop(self, hidden_output, final_output,targets,vector):
        print(targets)
        eo = self.error_outer(final_output, targets) # delta outer
        ei = self.error_hidden(hidden_output,eo,targets) # delta inner
        #ready to update values for olw.
        self.update_weights_outer(hidden_output, eo)
        print(vector)
        #print('before ',self.hlw[0])
        self.update_weights_inner(vector,ei)
    #    print('after ',self.hlw[0])

        exit(0)
        return(0)



    def update_weights_inner(self,vector,ei):
        print('nodes: ',len(ei))
        print('inputlen',len(vector))
        print('hlw', len(self.hlw[0]))

        for i in range(len(self.hlw)):
            for j in range(len(vector)):
            #    print(j)
            #    print(weights2[i])
                self.hlw[i][j] = self.hlw[i][j] - (self.eta *ei[i] * vector[j])


    # hidden_output is the activation from hidden
    def update_weights_outer(self, hidden_output, eo):
        for i in range(len(self.olw)):
            for j in range(len(hidden_output)):
                self.olw[i][j] = (self.olw[i][j]-hidden_output[j]*eo[i]*self.eta)


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

        #ei = [hidden_out[i]*(1-hidden_out[i] * self.olw[i]*eo[i] for i in range(len(hidden_out))]
        return ei

    # 4.14 from the book
    def error_outer(self, final_output, targets):
        eo = [(final_output[i]-targets[i]) for i in range(len(final_output))] # these are the deltas
        print('print eo:', eo)
        return(eo)


    def confusion(self, inputs, targets):
        print('To be implemented')

    # this is a linear function, might change this later, but ill let it stand for now.
    def activation_outer(self, weightedsum):
        return weightedsum

    # helper function for sigmoid on each neuron
    def sigmoid(self, weightedsum):
        print(self.beta)
        weightedsum = [1/(1+np.exp(-self.beta *weightedsum[i])) for i in range(len(weightedsum))]
        return weightedsum

    # helper funcion for calulating weightetsum for each neuron
    def matrix_hidden(self, vector):
        weightedsum = []
    #    print('here is nhidden', self.nhidden)
    #    print('here is hlw[0][0]',self.hlw[0][0])
    #    print(' MATRIX here is inputs', len(inputs[1]))

        for i in range(self.nhidden):
    #        print('in FOR', len(inputs))
        #    print('i',i)
            answ = 0
    #        print(len(self.hlw[0]))
    #        print(len(self.hlw[1]))
    #        print(len(self.hlw[11]))
    #        print(inputs[0])
            for j in range(len(self.hlw[0])-1):
            #    print('j',j)
                answ += self.hlw[i][j]*vector[j]
                answ += self.hlw[i][self.ninputs-1] * self.bias
            weightedsum.append(answ)
        #returns an array of the weighted sums.
        return weightedsum

    def matrix_outer(self,hidden_out):
        weightedsum = []
        print((len(hidden_out))-1)
        print(self.output_nr)
        print('len solw: ',len(self.olw[0]))
    #    print('here is nhidden', self.nhidden)
    #    print('here is hlw[0][0]',self.hlw[0][0])
    #    print(' MATRIX here is inputs', len(inputs[1]))
        #exit(0)

        for i in range(self.output_nr):
    #        print('in FOR', len(inputs))
        #    print('i',i)
            answ = 0
    #        print(len(self.hlw[0]))
    #        print(len(self.hlw[1]))
    #        print(len(self.hlw[11]))
    #        print(inputs[0])
            for j in range(len(self.olw[0])-1):
            #    print('j',j)
                answ += self.olw[i][j]*hidden_out[j]
                answ += self.olw[i][(len(hidden_out))-1] * self.bias
            weightedsum.append(answ)
        #returns an array of the weighted sums.
        return weightedsum
