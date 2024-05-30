#! /usr/bin/env python
# -*-coding: utf-8 -*-

__author__ = 'Gerald W. Liu'

'''FINA 5840 Homework 4
'''

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

'''Question 1
Write a generic Black-Scholes European option pricer which can take the input 
whether it is a Call (isCall = True) or put (isCall = False), 
the spot price (S), strike (K), 
risk-free rate (r), dividend yield (q), time-to-maturity (T), and volatility (sig). 
Use the function to price a call option with S=100, K=100, r=2%, q=3%, T=0.25, sig=0.2.
'''
def bs_price(is_call, S, K, r, q, T, sig):
    F = S * np.exp((r-q)*T)
    vol = sig * np.sqrt(T)
    d1 = np.log(F/K) / vol + 0.5 * vol
    d2 = d1 - vol
    sgn = 1 if is_call else -1

    return np.exp(-r*T) * sgn * (F*norm.cdf(sgn*d1) - K*norm.cdf(sgn*d2))

bs_params = {
    'is_call': True,
    'S': 100,
    'K': 100,
    'r': 0.02,
    'q': 0.03,
    'T': 0.25,
    'sig': 0.2
}

eu_c_price = bs_price(**bs_params)
print(eu_c_price)
# print(bs_price(True, 100, 100, 0.02, 0.03, 0.25, 0.2))

'''Question 2
Write a generic Binomial Tree Pricer so that it can take the input 
whether it is European (isEuropean = True) or American (isEuropean = False), 
whether it is a call or put (isCall = True/False), 
S, K, r, q, T, sig, n (number of steps).
'''
def bin_tree_price(is_eu, is_call, S, K, r, q, T, sig, n):
    dt = T / n
    u = np.exp(sig * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r-q)*dt) - d) / (u - d)
    sgn = 1 if is_call else -1

    S_vec = S * u**np.arange(n, -1, -1) * d**np.arange(n+1) # (n+1, )
    C_vec = np.maximum(sgn*(S_vec - K), 0) # (n+1, )

    for _ in range(n): # S_vec.shape[0] - 1
        S_vec = S_vec[:-1] / u
        C_vec = np.exp(-r*dt) * (p * C_vec[:-1] + (1-p) * C_vec[1:])
        if not is_eu:
            C_vec = np.maximum(C_vec, sgn*(S_vec - K))
    
    return C_vec[0]

'''Question 2 a.
Price the same option as Q1 but change it to American option.
'''
print(bin_tree_price(is_eu=False, n=100, **bs_params))
# print(bin_tree_price(False, True, 100, 100, 0.02, 0.03, 0.25, 0.2, 100))

'''Question 2 b.
Use the Binomial Tree for step size of 1 to 99 to find the difference between 
the Binomial price vs the Black-Scholes theoretical European Call price as Q1 
to produce the chart as our last slide in Lecture 6.
'''
n_range = np.arange(1, 100) # 1 to 99 steps
bin_prices = np.zeros(n_range.shape[0])
for i in n_range:
    bin_prices[i-1] = bin_tree_price(is_eu=False, n=i, **bs_params)

bin_error = bin_prices - eu_c_price

plt.plot(n_range, bin_error)
plt.xlabel('step size')
plt.ylabel('Binomial tree value vs BS')
plt.savefig('Bin-BS.png')
