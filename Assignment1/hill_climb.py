#!/usr/bin/python3
import time
import numpy as np
import random
from sys import exit
from functions import *


def hill_climb(cities,distances,number):

    shortest_route = list(range(number))
    random.shuffle(shortest_route)
    best_distance = calculate_distance(shortest_route,distances)

    for i in range(1000):
        random_genes = random.sample(list(shortest_route),2)
        shortest_route[random_genes[0]], shortest_route[random_genes[1]] = shortest_route[random_genes[1]], shortest_route[random_genes[0]]
        tmp = calculate_distance(shortest_route,distances)
        '''
        If the child is worse then parent, discard the mutation,and use parents genom again.
        '''
        if tmp > best_distance:
            shortest_route[random_genes[1]], shortest_route[random_genes[0]] = shortest_route[random_genes[0]], shortest_route[random_genes[1]]
        else:
            best_distance = tmp

    return shortest_route, best_distance

def hill_climb_start(cities, distances, low, high):

    # N = 10 cities
    route_array = list()
    distance_array = []
    start_time = time.time()

    for i in range(20):
        shortest_route, best_distance = hill_climb(cities,distances,low)
        route_array.append(shortest_route)
        distance_array.append(best_distance)

    end_time = time.time() - start_time
    print_answers(route_array, distance_array,cities,low,end_time)


    # N = 24 cities
    route_array.clear()
    distance_array.clear()
    start_time = time.time()

    for i in range(20):
        shortest_route, best_distance = hill_climb(cities,distances,high)
        route_array.append(shortest_route)
        distance_array.append(best_distance)

    end_time = time.time() - start_time
    print_answers(route_array, distance_array,cities,high,end_time)


def print_answers(route_array, distance_array,cities,number,time):

    shortest_route_index = np.argmin(distance_array)
    longest_route_index = np.argmax(distance_array)
    phenotype = geno_to_pheno(route_array[shortest_route_index],cities)
    best_city_distance = distance_array[shortest_route_index]
    worst_city_distance = distance_array[longest_route_index]
    mean_distances = sum(distance_array) / float(len(distance_array))
    standard_diviation = np.std(distance_array)


    print(f'Runtime for {number} cities visited: {time:2.4f} seconds with {20:d} random starting points and {1000:d} allele swaps.\n Length of best tour is: {best_city_distance:2.2f}.\n Length of worst tour is: {worst_city_distance:2.2f}.\n Length of average tour is: {mean_distances:2.2f}.\n Standard deviation is: {standard_diviation:2.2f}\n. Route of best tour is:{phenotype}.\n')


if __name__ == '__main__':
    cities, distances = reader("european_cities.csv")
    hill_climb_start(cities, distances, 10,24)
