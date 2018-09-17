#!/usr/bin/python3
'''
Code based on solutions for week 2 tasks. Thank you TA, you are the MVP
'''
import csv
from random import shuffle , sample, randint, random
import time
import numpy as np
from sys import exit


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
    #print('in parent selections')
    #print(population)
    #print(distances)
    #print(selection_size)
    random_selections = sample(population,selection_size)
    selection_distances = []

    for i in range(len(random_selections)):
        #calc = calculate_distance(random_selections[i],distances)
        #print(f'This is the {i} parent, total distance is : {calc}')
        selection_distances.append(calculate_distance(random_selections[i], distances))


    #print(f'random_selections before pop : {random_selections}')
    shortest_route_index = np.argmin(selection_distances)
    #print(f'parent_selection {shortest_route_index}')
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
    #print('here are the parents')
    #print(parent_one)
    #print(parent_two)
    return parent_one, parent_two


def mutate_citizen(citizen):
    #print('before')
    half = len(citizen) // 2 #rounded down
    #print('hello')
    start = randint(0, len(citizen)-half)
    stop = half + start
    #print(f'{start}:{stop}')
    #print(f'child before: {child}')
    citizen[start:stop] = citizen[start:stop][::-1]
    #print(f'child after: {child}')

    return citizen

def pmx(p1, p2,start,stop):

    '''
    Yes, I am so lazy that i convert from list to np.array on parents,
    and from np.array to list in children.
    '''
    child = [None]*len(p1) #init a empty list
    child[start:stop] = p1[start:stop] # copy a silce of parent
    parent_one = p1.tolist()
    parent_two = p2.tolist()

    for i,j in enumerate(parent_two[start:stop]):
        i += start
        if j not in child:
            while child[i] != None:
                i = parent_two.index(parent_one[i])
            child[i] = j

    # copy the rest from parent2
    for i,j in enumerate(child):
        if j == None:
            child[i] = p2[i]


    child = np.array(child)
    return child


def pair_pmx(p1,p2):
    half = len(p1) // 2 #rounded down
    start = randint(0, len(p1)-half)
    stop = half + start #now we only choose half of the genomes of the parent
    #print(f'pair_pmx: {start} : {stop}')
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

def genetic(cities, distances, population_size, max_generations, selection_size, mutation_rate,num_cities):

    population = make_population(population_size, num_cities)
    elites = []
    average_fitness = []
    generation = 0
    searched = 0
    #print(population)

#    print('in genetic')
    while generation < max_generations:
    #    print(f'{generation} vs {max_generations}')
        lenght = len(population)
    #    print(f'This is the population: {population}\n Size is : {lenght}')

        generation_distances = []
        #generation_distances =[calculate_distance(route,distances) for route in population]
    #    print(f'this is RIGHT before appending : {generation_distances}')

        for i in range(population_size):
            generation_distances.append(calculate_distance(population[i], distances))

    #    print(f'this is the generation_distances: {generation_distances}')
        shortest_route_index = np.argmin(generation_distances)
        shortest_route = generation_distances[shortest_route_index]
    #    print(f'here is the shortest route index so far : {shortest_route_index}. Value is : {shortest_route}')
        new_population = [population[shortest_route_index]] # save the best form previous gen

        elites.append(generation_distances[shortest_route_index]) # save for calculating avg/best/worst
    #    print(f'current population: {new_population}')
        #saving ~5% of the previous generation for next just to get some more randomnes

        for i in range(population_size//20):
            lucky_citizen = population[randint(0,population_size-1)]
            new_population.append(lucky_citizen)

    #    print(f'current population AFTER lucky_citizen: {new_population}')

        while len(new_population) < population_size:
            #print(distances)
            parent_one, parent_two = parent_selection(population, distances, selection_size)
            child_one, child_two = pair_pmx(parent_one, parent_two)
            #print(f'here is child_one : {child_one}')
            #print(f'here is child_two : {child_two}')

            new_population.append(child_one)
            new_population.append(child_two)

        #    tmp = len(new_population)
        #    print(f'size of new pop {tmp}')

        #print('population is created')

        if (max_generations/generations) > 0.8:
            mutation_rate = 0.025 # decrease mutation rate as we come within the last 20% runs
            for i in range(population_size):
                if random() < mutation_rate:
                    new_population[i] = mutate_citizen(new_population[i])


    #    print(f' ELITES   {elites}')

        average_fitness.append(np.mean(elites)) # save mean of elites to fitness as required
        #print(f'here is the avarage fitness so far: {average_fitness}')
        searched += len(population)
        population = new_population
        generation += 1
    #    print('CAME THROUGH WHOLE WHILE')



    #print('DONEEEEE\n\n')



    population_distances = []

    for i in range(population_size):
        population_distances.append(calculate_distance(population[i],distances))

    #print(f' This is final population: {population}')
    #print(f' This is the average_fitness_array: {average_fitness}')
    #print(f' This is population_distances: {population_distances}')
    #print(f' This is the elites: {elites}')


    return population, population_distances, average_fitness, searched
def calculate_distance(p,distances):

    sum = 0
    for i in range(len(p)-1):
        sum += distances[p[i],p[i+1]]

    sum += distances[p[-1],p[0]]
    return sum


def genetic_start(cities, distances, population_size, max_generations, num_cities,runs):

        mutation_rate = 0.05 # starting with this and changing
        selection_size = population_size//10 # give also weeker individuals a chance to make babies

        best_routes_all_run = []
        best_distance_all_run = []

        worst_distance_all_run = []

        average_distance_all_run = []
        std_all_runs = []

        planet_fit = []
        total_searched = 0
        start_time = time.time()

        for i in range(runs):

            p, d, f, s = genetic(cities, distances, population_size, max_generations, selection_size, mutation_rate, num_cities)

            best_route_index = np.argmin(d)   # best rounte index
            best_routes_all_run.append(p[best_route_index])
            best_distance_all_run.append(d[best_route_index]) # best distance


            worst_route_index = np.argmax(d)
            worst_distance_all_run.append(max(d)) # longest distance


            average_route_length = np.sum(d)/len(d)
            average_distance_all_run.append(average_route_length)

            std_all_runs.append(np.std(d))
            total_searched += s


        end_time = time.time() - start_time
        glb_best_route_index = np.argmin(best_distance_all_run) # get the best route
        global_best_dist = min(best_distance_all_run) # get the best distance
        #global_best_dist = best_distance_all_run[glb_best_route_index]
        #glb_worst_route_index = np.argmax(best_distance_all_run)

        global_worst_dist = max(worst_distance_all_run) # the worst distance of all


        global_average_dist = np.sum(average_distance_all_run)/len(average_distance_all_run)
        average_std = np.sum(std_all_runs)/len(std_all_runs)
        phenotype = geno_to_pheno(best_routes_all_run[best_route_index],cities)


        print(f'{num_cities} cities, with {population_size} population and {max_generations} generations:\n Length best tour: {global_best_dist:2.2f}.\n Length of worst tour: {global_worst_dist:2.2f}.\n Length of average tour: {global_average_dist:2.2f}.\n Standard diviation is: {average_std}.\n Route of the best tour is: {phenotype}\n Searched {total_searched} routes.\n Runtime: {end_time:2.4f}\n\n')

def geno_to_pheno(genotype, cities):
    pheno = []
    for i in genotype:
        pheno.append(cities[i])
    return pheno


if __name__ == '__main__':
        cities, distances = reader("european_cities.csv")
        # 10 cities
        num_cities = 10
        runs = 20
        all_stars_fitness = []
        population_size= [ 50, 100, 150]
        generations = 50




        for pop in population_size:
            genetic_start(cities, distances, pop, generations, num_cities,runs)

        N_cities = 24
        population_size= [100, 250, 600]
        generations = 100

        #print(population)
