from random import shuffle , sample, randint, random
import random
#from hill_climb import *
import numpy as np
from functions import *
import matplotlib.pyplot as plt
import time
from genetic import make_population, pair_pmx


def hill_climb_hybrid(cities,distances,geno):

    best_distance = calculate_distance(geno,distances)
    #geno = geno.tolist() # because im lazy
    #print(geno)


    for i in range(20):
        random_genes = random.sample(list(geno),2)
        #print(random_genes)
        geno[random_genes[0]], geno[random_genes[1]] = geno[random_genes[1]], geno[random_genes[0]]
        tmp = calculate_distance(geno,distances)
        '''
        If the child is worse then parent, discard the mutation,and use parents genom again.
        '''
        if tmp > best_distance:
            geno[random_genes[1]], geno[random_genes[0]] = geno[random_genes[0]], geno[random_genes[1]]
        else:
            best_distance = tmp

    return geno, best_distance

'''
Needs to be done with both Lamarican and Baldwinian
'''

def parent_selection_hybrid(population, distances, selection_size, string):

    random_selections = sample(population,selection_size) # contains all specimen with original genome
    hc_with_geno_saved = [] # contains all the modified parents after hill_climb
    distances_from_hc = []

    # run HC on our random selections
    for i in range(len(random_selections)):
        gen, dist = hill_climb_hybrid(cities, distances, random_selections[i])
        hc_with_geno_saved.append(gen)
        distances_from_hc.append(dist)

# now genomes and distances from originals are saved!
    selection_distances = []
    # if Lamarckian we use the new genomes for parents based on new fitness for selecting parents.
    if string == 'Lam':
        shortest_route_index = np.argmin(distances_from_hc)
        parent_one = hc_with_geno_saved[shortest_route_index]
        hc_with_geno_saved.pop(shortest_route_index)
        distances_from_hc.pop(shortest_route_index)
        shortest_route_index = np.argmin(distances_from_hc)
        parent_two = hc_with_geno_saved[shortest_route_index]

    # else , aka : Baldwanian we use the new fitness, but old genomes.
    else:
        shortest_route_index = np.argmin(distances_from_hc)
        parent_one = random_selections[shortest_route_index]
        random_selections.pop(shortest_route_index)
        distances_from_hc.pop(shortest_route_index)
        shortest_route_index = np.argmin(distances_from_hc)
        parent_two = random_selections[shortest_route_index]

    return parent_one, parent_two


def hybrid(cities, distances, population_size, max_generations, selection_size, mutation_rate,num_cities, string):

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


            parent_one, parent_two = parent_selection_hybrid(population, distances, selection_size, string)
            child_one, child_two = pair_pmx(parent_one, parent_two)

            new_population.append(child_one)
            new_population.append(child_two)

        if (max_generations/generations) > 0.8:
            mutation_rate = 0.025 # decrease mutation rate as we come within the last 20% runs
            for i in range(population_size):

                if random.uniform(0.0, 1.0) < mutation_rate:
        
                    new_population[i] = mutate_citizen(new_population[i])


        average_fitness.append(np.mean(elites)) # save mean of elites to fitness as required
        searched += len(population)
        population = new_population
        generation += 1

    population_distances = []

    for i in range(population_size):
        population_distances.append(calculate_distance(population[i],distances))

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

    return best_individual_distance, best_individual_route, worst_individual_distance, average_distance,\
    standard_diviation, average_fitness, searched



def hybrid_start(cities, distances, population_size, max_generations, num_cities,runs, string):


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

            best, route, worst, avg, std_div, fit, searched = hybrid(cities, distances, population_size, max_generations, selection_size, mutation_rate, num_cities,string)

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

def plotter(fit, pop_size, gens, name):
    line1 = fit[0]
    line2 = fit[1]

    plt.title = ("Average fitness of each generation")
    plt.xlabel = ("Generations")
    plt.ylabel = ("Route length")
    plt.plot(line1, label='Lamarckian', color = 'C0')
    plt.plot(line2, label='Baldwinian ', color = 'C1')



    plt.legend()
    plt.savefig(name, dpi= 'figure', format ='png')
    plt.show()

    return 0



if __name__ == '__main__':
        cities, distances = reader("european_cities.csv")

        # 10 cities
        num_cities = 10
        runs = 20
        all_stars_fitness = []
        population_size= 100
        generations = 50

        fit = hybrid_start(cities, distances, population_size, generations, num_cities,runs, "Lam")
        all_stars_fitness.append(fit)
        fit = hybrid_start(cities, distances, population_size, generations, num_cities,runs, "Bal")
        all_stars_fitness.append(fit)
        plotter(all_stars_fitness, population_size, generations,'hybrid_10.png')

        exit(0)
        np.empty(all_stars_fitness)

        # 10 cities
        num_cities = 24
        runs = 20
        all_stars_fitness = []
        population_size = 200
        generations = 100

        fit = hybrid_start(cities, distances, population_size, generations, num_cities,runs, "Lam")
        all_stars_fitness.append(fit)
        fit = hybrid_start(cities, distances, population_size, generations, num_cities,runs, "Bal")
        plotter(all_stars_fitness, population_size, generations,'hybrid_24.png')
