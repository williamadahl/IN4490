#!/usr/bin/env Python3
'''
    This file will read in data and start your mlp network.
    You can leave this file mostly untouched and do your
    mlp implementation in mlp.py.
'''
# Feel free to use numpy in your MLP if you like to.
import numpy as np
import mlp
import os


def shift(movements,foldsize):
    # print(type(movements))
    movements = movements.tolist()
    movements = movements[-foldsize:] + movements[:-foldsize]
    movements = np.array(movements)
    #print(len(movements))




filename = '../data/movements_day1-3.dat'

movements = np.loadtxt(filename,delimiter='\t')

# Subtract arithmetic mean for each sensor. We only care about how it varies:
movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)

# Find maximum absolute value:
imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                          np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                          axis=0 ).max(axis=0)

# Divide by imax, values should now be between -1,1
movements[:,:40] = movements[:,:40]/imax[:40]

# Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0]
target = np.zeros((np.shape(movements)[0],8));
for x in range(1,9):
    indices = np.where(movements[:,40]==x)
    target[indices,x-1] = 1

# Randomly order the data
order = list(range(np.shape(movements)[0]))
np.random.shuffle(order)
movements = movements[order,:]
target = target[order,:]

folds = 5

foldsize = len(movements) // folds
#print(foldsize)
#exit(0)

test_fold_start_index = 0
test_fold_end_index = foldsize

validation_fold_start_index = foldsize
validation_fold_stop_index = foldsize*2

training_fold_start_index = validation_fold_stop_index

#print('TEST: ', test_fold_start_index, '-', test_fold_end_index)
#print('VALID: ',validation_fold_start_index, '-', validation_fold_stop_index)
#print('TRAINING: ',training_fold_start_index, '-',len(movements))

hidden = 12

train = movements[::2,0:40]
train_targets = target[::2]
#print(len(train[0]))
#print(len(train_targets[0]))

score_array = []
# Initialize the network:
#net = mlp.mlp()
net = mlp.mlp(train, train_targets, hidden)


for i in range(folds):

    test_data = movements[0:foldsize,0:40]
    test_targets = target[0:foldsize]

    valid_data = movements[foldsize:foldsize*2,0:40]
    valid_targets = target[foldsize:foldsize*2]

    training_data = movements[training_fold_start_index:,0:40]
    training_target = target[training_fold_start_index:]

    net.earlystopping(training_data, training_target, valid_data, valid_targets)
    dummy = net.confusion(test_data,test_targets)

    shift(movements, foldsize)
    shift(target, foldsize)
    score_array.append(dummy)


average = np.average(score_array)
stddiv =  np.std(score_array)
print('\n-------------------------------')
print(f'Average correctness: {average:2.2f}%')
print(f'Standard deviation: {stddiv:2.2f}')
