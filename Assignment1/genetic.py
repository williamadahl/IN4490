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
        population.append(citizen)

    return population


def parent_selection(population, distances, selection_size):

    random_selections = sample(population,selection_size)
    selection_distances = []

    for i in range(len(random_selections)):
        selection_distances.append(calculate_distance(random_selections[i], distances))


    shortest_route_index = np.argmin(selection_distances)
    parent_one = random_selections[shortest_route_index]
    random_selections.pop(shortest_route_index)
    selection_distances.pop(shortest_route_index)
    shortest_route_index = np.argmin(selection_distances)
    parent_two = random_selections[shortest_route_index]
    return parent_one, parent_two


def mutate_citizen(citizen):
    half = len(citizen) // 2 #rounded down
    start = randint(0, len(citizen)-half)
    stop = half + start
    citizen[start:stop] = citizen[start:stop][::-1]

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

    while generation < max_generations:
        lenght = len(population)
        generation_distances = []

        for i in range(population_size):
            generation_distances.append(calculate_distance(population[i], distances))

        shortest_route_index = np.argmin(generation_distances)
        shortest_route = generation_distances[shortest_route_index]
        new_population = [population[shortest_route_index]] # save the best form previous gen

        elites.append(generation_distances[shortest_route_index]) # save for calculating avg/best/worst

        #saving ~5% of the previous generation for next just to get some more randomnes
        for i in range(population_size//20):
            lucky_citizen = population[randint(0,population_size-1)]
            new_population.append(lucky_citizen)


        while len(new_population) < population_size:

            parent_one, parent_two = parent_selection(population, distances, selection_size)
            child_one, child_two = pair_pmx(parent_one, parent_two)

            new_population.append(child_one)
            new_population.append(child_two)

        if (max_generations/generations) > 0.8:
            mutation_rate = 0.025 # decrease mutation rate as we come within the last 20% runs
            for i in range(population_size):
                if random() < mutation_rate:
                    new_population[i] = mutate_citizen(new_population[i])



        average_fitness.append(np.mean(elites)) # save mean of elites to fitness as required
        searched += len(population)
        population = new_population
        generation += 1



    population_distances = []

    for i in range(population_size):
        population_distances.append(calculate_distance(population[i],distances))

    runs += 1

    #The best individual of last generation
    best_individual_index = np.argmin(population_distances)
    best_individual_distance = population_distances[best_individual_index]
    best_individual_route = population[best_individual_index]

    #The worst individual of last generation
    worst_individual_index = np.argmax(population_distances)
    worst_individual_distance = population_distances[worst_individual_index]

    #Average distance last generation
    average_distance = np.sum(population_distances)/len(population_distances)

    #Standard_diviation
    standard_diviation = np.std(population_distances)


    #exit(0)
    return best_individual_distance, best_individual_route, worst_individual_distance, average_distance,\
    standard_diviation, average_fitness, searched

#    return population, population_distances, average_fitness, searched
def calculate_distance(p,distances):

    sum = 0
    for i in range(len(p)-1):
        sum += distances[p[i],p[i+1]]

    sum += distances[p[-1],p[0]]
    return sum


def genetic_start(cities, distances, population_size, max_generations, num_cities,runs):


        mutation_rate = 0.05 # starting with this and changing
        selection_size = population_size//10 # give also weeker individuals a chance to make babies
        best_distance_all_run = []
        best_routes_all_run = []
        worst_distance_all_run = []
        average_distance_all_run = []
        std_all_runs = []
        planet_fittnes = np.zeros((runs, generations))
        total_searched = 0
        start_time = time.time()

        for i in range(runs):

            best, route, worst, avg, std_div, fit, searched = genetic(cities, distances, population_size, max_generations, selection_size, mutation_rate, num_cities)

            best_distance_all_run.append(best)
            best_routes_all_run.append(route)
            worst_distance_all_run.append(worst)
            average_distance_all_run.append(avg)
            std_all_runs.append(std_div)
            planet_fittnes[i] = fit
            total_searched += searched

        end_time = time.time() - start_time

        glb_best_route_index = np.argmin(best_distance_all_run)
        glb_best_distance = best_distance_all_run[glb_best_route_index]
        glb_best_route = best_routes_all_run[glb_best_route_index]

        glb_worst_route_index = np.argmax(worst_distance_all_run)
        glb_worst_dist = worst_distance_all_run[glb_worst_route_index]
        glb_average_dist = np.sum(average_distance_all_run)/len(average_distance_all_run)
        glb_std = np.sum(std_all_runs)/len(std_all_runs)
        phenotype = geno_to_pheno(glb_best_route, cities)
        ret_fit = planet_fittnes.mean(axis=0)

        print(f'{num_cities} cities, with {population_size} population and {max_generations} generations:\n Length best tour: {glb_best_distance:2.2f}.\n Length of worst tour: {glb_worst_dist:2.2f}.\n Length of average tour: {glb_average_dist:2.2f}.\n Standard diviation is: {glb_std}.\n Route of the best tour is: {phenotype}\n Searched {total_searched} routes.\n Runtime: {end_time:2.4f}\n\n')

        return ret_fit

def geno_to_pheno(genotype, cities):
    pheno = []
    for i in genotype:
        pheno.append(cities[i])
    return pheno

def plotter(fit, pop_size, gens):




if __name__ == '__main__':
        cities, distances = reader("european_cities.csv")
        # 10 cities
        num_cities = 10
        runs = 20
        all_stars_fitness = []
        population_size= [ 50, 100, 150]
        generations = 50


        for pop in population_size:
            fit = genetic_start(cities, distances, pop, generations, num_cities,runs)
            all_stars_fitness.append(fit)

        plotter(all_stars_fitness, population_size, generations)


        num_cities = 24
        population_size = [100, 200, 300]
        generations = 100

        for pop in population_size:
            fit = genetic_start(cities, distances, pop, generations, num_cities,runs)
            all_stars_fitness.append(fit)
