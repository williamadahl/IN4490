#!/usr/bin/python3
'''
Code based on solutions for week 2 tasks. Thank you TA, you are the MVP
'''
import csv
import random
import time
import numpy as np 




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
