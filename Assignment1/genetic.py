#!/usr/bin/python3
'''
Code based on solutions for week 2 tasks. Thank you TA, you are the MVP
'''
import csv
import random
import time
import numpy as np




def pmx(p1, p2,start,stop):
    child = [None]*len(p1) #init a empty list
    child[start:stop] = p1[start:stop] # copy a silce of parent

    # map the slice from p2 into child using the indices from p1
    print(child)
    for i,j in enumerate(p2[start:stop]):
        i += start
        # now we check if there are genes in p2 substring that are not in the child yet.
        if j not in child:
            print('this is j '+str(j))
            while child[i] != None:
                print('this is i: ' + str(i))
                i = p2.index(p1[i])
                print('this is  new i: ' + str(i))
            child[i] = j

    print(child)
    # copy the rest from parent2
    for i,j in enumerate(child):
        if j == None:
            child[i] = p2[i]

    return child


def pair_pmx(p1,p2):
    half = len(p1) // 2 #rounded down
    start = rand.randint(0, len(p1)-half)
    stop = half + start #now we only choose half of the genomes of the parent
    return pmx(p1,p2,start,stop), pmx(p2,p1,start,stop) # call to pmx and create the crossover children.


def reader(filename):

    with open(filename, "r") as f:
        data = list(csv.reader(f, delimiter=';'))

    cities = []
    distances = np.zeros((len(data)-1,len(data)-1))
    city_line = 0

    for line in data:
        if city_line == 0:
             for i in range(len(line)):
                 cities.append(line[i])
        else:
            for i in range(len(line)):
                distances[city_line - 1,i] = line[i]

        city_line += 1
    return cities, distances



if __name__ == '__main__':
        cities, distances = reader("european_cities.csv")
        pmx_start()
