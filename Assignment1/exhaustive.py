import csv
from itertools import permutations
import numpy as np
import sys


def exhaustive(cities, distances,number):

    shortest_route = np.zeros(number)
    best_distance = sys.maxsize
    perm = permutations(number)


    print(str(perm))

    # print(len(shortest_route))
    return 5


def exhautive_plotter():
    return 0

def exhaustive_start():
    return 0



def reader(filename):

    with open(filename, "r") as f:
        data = list(csv.reader(f, delimiter=';'))

    cities = []
    distances = np.zeros((len(data)-1,len(data)-1))
    print(str(distances))
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
    test = exhaustive(distances, cities, 6)
