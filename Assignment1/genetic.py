#!/usr/bin/python3
'''
Code based on solutions for week 2 tasks. Thank you TA, you are the MVP
'''
import csv
from random import shuffle , sample, randint
import time
import numpy as np


def make_population(size,N):

    population = []
    for i in range(size):
        citizen = np.random.permutation(N)
    #    print(citizen)
        population.append(citizen)

    #for i in range(len(population)):
    #    print(f'Here is citizen number {i}: {population[i]}')
    return population


def parent_selection(population, distances, selection_size):

    random_selections = sample(population,selection_size)
    selection_distances = []

    for i in range(len(random_selections)):
        calc = calculate_distance(random_selections[i],distances)
        #print(f'This is the {i} parent, total distance is : {calc}')
        selection_distances.append(calculate_distance(random_selections[i], distances))


    #print(f'random_selections before pop : {random_selections}')
    shortest_route_index = np.argmin(selection_distances)
    #print(f'shortest route index : {shortest_route_index}')
    parent_one = random_selections[shortest_route_index]
    #print(parent_one)
    random_selections.pop(shortest_route_index)
    selection_distances.pop(shortest_route_index)
    #print(f'random_selections after pop : {random_selections}')
    #print(f'distances after pop : {selection_distances}')
    shortest_route_index = np.argmin(selection_distances)
    parent_two = random_selections[shortest_route_index]
    #print(parent_two)

    return parent_one, parent_two


def pmx(p1, p2,start,stop):
    child = [None]*len(p1) #init a empty list
    child[start:stop] = p1[start:stop] # copy a silce of parent

    # map the slice from p2 into child using the indices from p1
    print(child)
    for i,j in enumerate(p2[start:stop]):
        i += start
        # now we check if there are genes in p2 substring that are not in the child yet.
        if j not in child:
        #    print('this is j '+str(j))
            while child[i] != None:
        #        print('this is i: ' + str(i))
                i = p2.index(p1[i])
        #        print('this is  new i: ' + str(i))
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

def calculate_distance(p,distances):

    sum = 0
    for i in range(len(p)-1):
        sum += distances[p[i],p[i+1]]

    sum += distances[p[-1],p[0]]
    return sum


if __name__ == '__main__':
        cities, distances = reader("european_cities.csv")
        # 10 cities

        N = 10 # 10 cities
        generations = 50
        population_size = 20
        selection_size = population_size//4

        population = make_population(population_size,N)
        #print(population)
        p1,p2 = parent_selection(population, distances, selection_size)
        
