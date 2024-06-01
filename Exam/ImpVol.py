__author__ = 'Gerald W. Liu'

import numpy as np
from scipy.stats import norm
from functools import partial

def bs_price(is_call, S, K, r, q, T, sig):
    F = S * np.exp((r-q)*T)
    vol = sig * np.sqrt(T)
    d1 = np.log(F/K) / vol + 0.5 * vol
    d2 = d1 - vol
    sgn = 1 if is_call else -1

    return np.exp(-r*T) * sgn * (F*norm.cdf(sgn*d1) - K*norm.cdf(sgn*d2))

def vega(S, K, r, q, T, sig):
    F = S * np.exp((r-q)*T)
    F0 = S * np.exp(-q*T)
    vol = sig * np.sqrt(T)
    d1 = np.log(F/K) / vol + 0.5 * vol

    return F0 * norm.pdf(d1) * np.sqrt(T)

def iv_bisec(price, is_call, S, K, r, q, T):
    tol = 1e-10
    sig_lo = 1e-4
    sig_hi = 0.0
    incr = 0.2
    i = 0
    i_max = 10000
    bs_price_partial = partial(bs_price, is_call, S, K, r, q, T)

    while bs_price_partial(sig_hi) - price < 0:
        sig_hi += incr

    while i < i_max:
        sig_temp = (sig_lo + sig_hi) / 2
        if bs_price_partial(sig_temp) - price > 0:
            sig_hi = sig_temp
        else:
            sig_lo = sig_temp
        
        if abs(bs_price_partial(sig_temp) - price) < tol:
            break
        i += 1
    
    return sig_temp

def iv_newton(price, is_call, S, K, r, q, T):
    tol = 1e-10
    sig = 1 # initial guess
    i = 0
    i_max = 10000
    bs_price_partial = partial(bs_price, is_call, S, K, r, q, T)
    vega_partial = partial(vega, S, K, r, q, T)

    while i < i_max:
        sig_temp = sig
        sig -= (bs_price_partial(sig_temp) - price) / vega_partial(sig_temp)
        if abs(bs_price_partial(sig_temp) - price) < tol:
            break

        i += 1
    
    return sig_temp

# test
bs_params = {
    'is_call': True,
    'S': 1,
    'K': 1,
    'r': 0,
    'q': -0.05,
    'T': 1
}
p1 = bs_price(**bs_params, sig=0.4)

print('IV by bisection method', iv_bisec(p1, **bs_params))
print('IV by newton method', iv_newton(p1, **bs_params))
