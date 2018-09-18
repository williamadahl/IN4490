import csv
import numpy as np


'''
calculate distance from starting point, to next visited, and traveling form visited to
next destination acording to the permutation.
In the end adding the distance form last city visited, back to starting city
'''

def calculate_distance(p,distances):

    sum = 0
    for i in range(len(p)-1):
        sum += distances[p[i],p[i+1]]

    sum += distances[p[-1],p[0]]
    return sum

def geno_to_pheno(genotype, cities):
    pheno = []
    for i in genotype:
        pheno.append(cities[i])
    return pheno


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
