#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return -x**4 + 2*x**3 + 2*x**2 -x

def df(x):
    return -4*x**3 + 6*x**2 + 4*x -1

#1b

x = np.linspace(-2,3,100)
plt.plot(x,f(x),label="f(x)")
plt.plot(x,df(x),label="df(x)")
plt.legend()
plt.savefig("1bc")
plt.show()
