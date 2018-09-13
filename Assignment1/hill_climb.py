#!/usr/bin/python3
import csv
import time
import numpy as np
import random


def hill_climb(cities,distances,number):
    shortest_route = np.arange(number)
    random.shuffle(shortest_route)

    best_distance = calculate_distance(shortest_route,distances)
    # Leting the swaping run 2^10 times.
    for i in range(2**10):
        random_genes = random.sample(list(shortest_route),2)
    #    print(f'{random_tuples}')
        shortest_route[random_genes[0]], shortest_route[random_genes[1]] = shortest_route[random_genes[1]], shortest_route[random_genes[0]]
        tmp = calculate_distance(shortest_route,distances)
        '''
        If the child is worse than parent, discard the mutation,and use parents genom again.
        '''
        if tmp > best_distance:
            shortest_route[random_genes[1]], shortest_route[random_genes[0]] = shortest_route[random_genes[0]], shortest_route[random_genes[1]]

    return shortest_route,best_distance
    #    print(str(random_genes))
    #    print(f'{random_tuples}')

def hill_climb_start(cities, distances, low, high):

    shortest_route, best_distance = hill_climb(cities,distances,low)
    longest_route, worst_distance = best_route, shortest_route

    for i in range(20):
        tmp_shortest_route, tmp_best_distance = hill_climb(cities,distances,low)
        



    best_route, shortest_route = hill_climb(cities,distances,high)



def reader(filename):
    with open(filename,'r') as f:
        data = list(csv.reader(f, delimiter=';'))

    cities = []
    distances = np.zeros((len(data)-1, len(data)-1))
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

def geno_to_pheno(genotype, cities):
    pheno = []
    for i in genotype:
        pheno.append(cities[i])
    return pheno

def calculate_distance(p,distances):

    sum = 0
    for i in range(len(p)-1):
        sum += distances[p[i],p[i+1]]

    sum += distances[p[-1],p[0]]
    return sum

if __name__ == '__main__':
    cities, distances = reader("european_cities.csv")
    hill_climb_start(cities, distances, 10,24)
