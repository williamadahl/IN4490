#!usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import random as rand

def f(x):
    return -x**4 + 2*x**3 + 2*x**2 -x

def df(x):
    return -4*x**3 + 6*x**2 + 4*x -1

def gradient_acent(x, step_size, precision):
    dx = step_size * df(x)  # this is first calculation of random point. x = start.
    iterations = 0
    while abs(dx) > precision:
        iterations += 1
        plt.plot(x,f(x), color = "red", marker = "s", markersize=3)
        x = x + dx    # calcualate next
        dx = step_size * df(x) # calculate next
    return x, f(x), iterations


def gradient_acent_plotter(start, stop, steps):
    x = np.linspace(start, stop, steps) # list of values
    plt.plot(x,f(x))    #plot the graph first
    start_point = rand.uniform(start,stop) # choose random starting point
    answ = gradient_acent(start_point, 0.1, 0.0001)
    plt.plot(answ[0],answ[1], color="yellow", marker="*", markersize=10) #mark the solution with a star
    plt.savefig("gradient_acent.eps", format='eps') # save in .eps for great quality
    print('start_point: '+ str(start_point) +  "\n"'max: ' + str(answ[1]) +"\n"+'iterations: '+ str(answ[2])) # some info in the terminal
    plt.show()



if __name__ == '__main__':
    gradient_acent_plotter(-2,3,100)

'''
The choice of starting point and stepsize can have a lot to say for the performance. The most important is the step size since it will determine how many iterations we need until we fullfill the precision we want. start point can also affect performance, depending on if we are near a local or global max.

I am not sure if there is one point where we would not find a local max, but maybe if we started in the global minimum or critical point. the abs(dx) would be 0, and therefore the algorithm would stop. Gradient acent/decent therefore has no guarantee for finding local max / min. This is correct!

Also, stepsize to large might bounce over solutions!
Too low yields poor performance. 

This is a continous optimization
'''
