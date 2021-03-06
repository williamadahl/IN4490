import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from copy import deepcopy



class mlp:
    def __init__(self, inputs, targets, nhidden):
        self.beta = 1
        self.bias = 1
        self.eta = 0.1
        self.momentum = 0.0
        ninputs = len(inputs[0]) + 1
        noutput = len(targets[0])
        self.output_nr = noutput
        self.nhidden = nhidden
        self.ninputs = ninputs

        self.hlw = np.random.uniform(-1,1,(nhidden,ninputs))
        self.olw = np.random.uniform(-1,1,(noutput, nhidden + 1))

    def earlystopping(self, inputs, targets, valid, validtargets):

        best_guess = 0
        strikes = 0

        while 1:

            if strikes == 10:
                self.unlearn(best_innerweights, best_outerweights)
                break

            self.train(inputs, targets, 1)
            correct_guesses = 0

            for i in range(len(valid)):
                hidden , output = self.forward(valid[i]) # the testing
                index_validtargets = np.argmax(validtargets[i])
                index_output = np.argmax(output)
                if(index_validtargets == index_output):
                    correct_guesses += 1

            if correct_guesses >= best_guess:
                strikes = 0
                best_guess = correct_guesses
                best_innerweights = deepcopy(self.hlw)
                best_outerweights = deepcopy(self.olw)
            else:
                strikes += 1

            print('Best guess :',best_guess, '\tCurrent :',correct_guesses)
            print('Number of strikes: ', strikes)

            # removed this condition to improve training.
            #if (correct_guesses >= round(len(valid)*0.9)):
            #    break


    def train(self, inputs, targets, iterations):
        for it in range(iterations):
            for i in range(len(inputs)):
                #forward
                hidden_out, final_output = self.forward(inputs[i])
                #backward
                self.back_prop(hidden_out,final_output,targets[i],inputs[i])


    def forward(self, vector):
        hidden_out = self.matrix_hidden(vector)
        hidden_out = self.sigmoid(hidden_out)
        final_output = self.matrix_outer(hidden_out)
        final_output = self.activation_outer(final_output)
        return hidden_out, final_output

    def back_prop(self, hidden_output, final_output,targets,vector):
        eo = self.error_outer(final_output, targets)
        ei = self.error_hidden(hidden_output,eo,targets)

        self.update_weights_outer(hidden_output, eo)
        self.update_weights_inner(vector,ei)

    def update_weights_inner(self,vector,ei):
        for i in range(len(self.hlw)):
            for j in range(len(vector)):
                self.hlw[i][j] = self.hlw[i][j] - (self.eta *ei[i] * vector[j])

        for i in range(len(self.hlw)):
            self.hlw[i][len(self.hlw[0])-1] = self.hlw[i][len(self.hlw[0])-1] - (self.eta*ei[i]*self.bias)

    # Hidden_output is the activation from hidden
    def update_weights_outer(self, hidden_output, eo):
        for i in range(len(self.olw)):
            for j in range(len(hidden_output)):
                self.olw[i][j] = (self.olw[i][j]-hidden_output[j]*eo[i]*self.eta)

        for i in range(len(self.olw)):
            self.olw[i][len(self.olw[0])-1] = self.olw[i][len(self.olw[0])-1] - (self.eta*eo[i]*self.bias)


    # 4.9 from the book
    def error_hidden(self, hidden_output, eo,targets):
        ei = []
        for i in range(len(hidden_output)):
            answ = 0
            for j in range(len(targets)):
                answ += self.olw[j][i]*eo[j]
            ei.append(answ)
        return ei

    # 4.14 from the book
    def error_outer(self, final_output, targets):
        eo = [(final_output[i]-targets[i]) for i in range(len(final_output))] # these are the deltas
        return(eo)


    def confusion(self, inputs, targets, fold):

        actual = [0]*len(inputs)
        predicted = [0]*len(inputs)

        for i in range(len(inputs)):
            hidden , output = self.forward(inputs[i])
            index_out = np.argmax(output)
            predicted[i] += index_out

        for i in range(len(targets)):
            index = np.argmax(targets[i])
            actual[i] += index

        result = confusion_matrix(actual, predicted)
        score = accuracy_score(actual, predicted, normalize=True, sample_weight=None)

        if (fold == -1):
            print(f'\n\n-------------------------------\nNumber of hidden nodes: {self.nhidden}\n-------------------------------\nConfusion matrix:\n\n{result}')
            score = score*100
            print(f'\n-------------------------------\nCorrectness score: {score:2.2f}%\n-------------------------------')
        else:
            print(f'\n\n-------------------------------\nFold number: {fold}\n-------------------------------\nConfusion matrix:\n\n{result}')
            score = score*100
            print(f'\n-------------------------------\nCorrectness score: {score:2.2f}%\n-------------------------------')
        return score


    def activation_outer(self, weightedsum):
        return weightedsum

    # Helper function for sigmoid on each neuron
    def sigmoid(self, weightedsum):
        weightedsum = [1/(1+np.exp(-self.beta *weightedsum[i])) for i in range(len(weightedsum))]
        return weightedsum

    # Helper funcion for calulating weightetsum for each neuron
    def matrix_hidden(self, vector):
        weightedsum = []
        for i in range(self.nhidden):
            answ = 0
            for j in range(len(self.hlw[0])-1):
                answ += self.hlw[i][j]*vector[j]
            answ += self.hlw[i][self.ninputs-1] * self.bias
            weightedsum.append(answ)
        return weightedsum

    def matrix_outer(self,hidden_out):
        weightedsum = []
        for i in range(self.output_nr):
            answ = 0
            for j in range(len(self.olw[0])-1):
                answ += self.olw[i][j]*hidden_out[j]
            answ += self.olw[i][(len(hidden_out))-1] * self.bias
            weightedsum.append(answ)
        return weightedsum

    # "You must unlearn what you have learnt" -Grand Master Yoda
    def unlearn(self, best_innerweights, best_outerweights):
        self.hlw = deepcopy(best_innerweights)
        self.olw = deepcopy(best_outerweights)
