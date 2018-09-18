#!/usr/bin/python

import csv
from itertools import permutations
import numpy as np
import sys
import time
from functions import *


def exhaustive(cities, distances,number):

    perm_tuples = np.arange(number)
    shortest_route = np.zeros(number)
    best_distance = sys.maxsize
    perm = permutations(perm_tuples)

    for p in perm:
        tmp = calculate_distance(p,distances)
        if tmp < best_distance:
            best_distance = tmp
            shortest_route = p

    return shortest_route, best_distance


def exhaustive_start(cities, distances,N):

    for i in range(6,N+1):
        start_time = time.time()
        shortest_route, best_distance = exhaustive(cities,distances,i)
        end_time = time.time() - start_time
        print(f'Runtime for {i} cities is: {end_time:2.5f} seconds.\nMinimum distance is: {best_distance:2.2f}.')
        if i == 10:
            phenotype = geno_to_pheno(shortest_route,cities)
            print(f'Shortest tour for {i} cities:\n{phenotype}')


if __name__ == '__main__':
    cities, distances = reader("european_cities.csv")
    exhaustive_start(cities, distances, 10)
