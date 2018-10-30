#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import random

def f(x):
    return -x**4 + 2*x**3 + 2*x**2 -x


def brute_force(start, stop, step_size):
    x = start
    best = (x,f(x)) # these are the valuese on x and y axis
    while x < stop:
        y = f(x) # calulate y value
        plt.plot(x, f(x), color = "red", marker = "s" ,markersize = 3) # mark values on graph
        if y > best[1]: # if it's higher than initial, update best
            best = (x,y)

        x = x + step_size # increment for exhastive search
    return best



def plot_brute_force(start, stop, steps):
    x = np.linspace(start,stop,steps)
    plt.plot(x,f(x))
    start_point = random.uniform(start, stop)
    answ = brute_force(start, stop, 0.5)
    plt.plot(answ[0],answ[1], color = 'yellow', marker = '*', markersize = 10)
    plt.savefig('week1_1d.eps', format = 'eps')
    print('start_point: '+ str(start_point) +  "\n"'max: ' + str(answ[1])) # some info in the terminal
    plt.show()


if __name__ == '__main__':
    plot_brute_force(-2,3,100)

'''
1e)

Greedy search would not change the delta value and therefore be less precise than the gradient max. it would also not get stuck at the critical point where the gradient accent might endself. Greedy also mostly get the local optimum, not the global optimum. We are dependent on a good granularity (small step size).

For hill climbing we can encounter a plateu, and the algo will not determine what way it should choose for improvement, and might wander in a direction that will never lead to an improvement. Also, hill climbing in general is slower than the other optimization methods we use.


We might atleast get an answer from hill climbing and greedy search, whereas the gradient accent might not when we are in a plateu.

Hill climber: something better than where we are at.
Greedy : choose the best of two ways

Hill climber has a smaller chance of getting stuck at a local min.
One point where they might differ is for x = 0.5. This gives the hill climber a 50/50 chance of going left or right.


1f)

We could choose more than one startig point, and store the different local maxes in a list, and from that list choose the maximal value. Aka random seeding.



1g)

Depends on the step size on exhaustive search. IF we have a large step size we compute faster, but the result is less precise. With simulated annealing, we need to do slow search to make sure to not overshoot the global max.

Since the problem is simple and the stepsize is large, the brute force is better for maximizing than the simulated annealing. In GERERAL tho, and with more complex problems we CANNOT rely on brute force. Then sumulated annealing will give us an acceptable answer.

Terms:

Exploration: is usually taking sub-optimal steps

'''
