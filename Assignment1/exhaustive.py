import csv
from itertools import permutations
import numpy as np
import sys
import time


def exhaustive(cities, distances,number):

    perm_tuples = np.arange(number)
    shortest_route = np.zeros(number)
    best_distance = sys.maxsize
    perm = permutations(perm_tuples)
    #print(distances)

    for p in perm:
        tmp = calculate_distance(p,distances)
        if tmp < best_distance:
            best_distance = tmp
            shortest_route = p


    return shortest_route, best_distance



'''
calculate distance from starting point, to next visited, and traveling form visited to
next destination acording to the permutation.
In the end adding the distance form last city visited, back to starting city
'''
def calculate_distance(p,distances):

    sum = 0
    for i in range(len(p)-1):
        sum += distances[p[i],p[i+1]]
        print('inside for' + str(distances[p[i],p[i+1]]))


    sum += distances[p[-1],p[0]]

    return sum

def exhautive_plotter():
    return 0

def exhaustive_start():
    return 0



def reader(filename):

    with open(filename, "r") as f:
        data = list(csv.reader(f, delimiter=';'))

    cities = []
    distances = np.zeros((len(data)-1,len(data)-1))
    #print(str(distances))
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
    #print(str(cities))
    #print(str(distances))
    test,tull = exhaustive(cities, distances, 4)
