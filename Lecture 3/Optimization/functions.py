#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from datetime import datetime
from numpy.random import multivariate_normal, randint
import yfinance as yf

import scipy.stats as scs
import scipy.optimize as sco
import scipy.interpolate as sci

from matplotlib.ticker import FuncFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt


def print_statistics(array):
    ''' Prints selected statistics.
    
    Parameters
    ==========
    array: ndarray object to generate statistics on
    '''
    sta = scs.describe(array)
    print("%14s %15s" % ('statistic', 'value'))
    print(30 * "-")
    print("%14s %15.5f" % ('size', sta[0]))
    print("%14s %15.5f" % ('min', sta[1][0]))
    print("%14s %15.5f" % ('max', sta[1][1]))
    print("%14s %15.5f" % ('mean', sta[2]))
    print("%14s %15.5f" % ('std', np.sqrt(sta[3])))
    print("%14s %15.5f" % ('skew', sta[4]))
    print("%14s %15.5f" % ('kurtosis', sta[5]))
    
    
def normality_test(array):
    
    print("Skew of data set %14.3f" % scs.skew(array))
    print("Skew test p-value %14.3f" % scs.skewtest(array)[1])
    print("Kurt of data set %14.3f" % scs.kurtosis(array))
    print("Kurt test p-value %14.3f" % scs.kurtosistest(array)[1])
    print("Norm test p-value %14.3f" % scs.normaltest(array)[1])

def rand_weights(n):
    k = np.random.rand(n)
    return k / sum(k)

def portf_mean(w, mean):
    mu = w.dot(mean)
    return mu
    
def portf_var(w, cov_matrix):
    
    sigma = np.diagonal(w.dot(cov_matrix).dot(w.T))
    return sigma    

def returns_plot(case1, case2, case3, case4, case5):
    
    fig = plt.figure(figsize=(15,8))

    plt.subplot(231)
    plt.plot(case1, alpha=.8);
    plt.xlabel('time')
    plt.ylabel('returns')
    plt.title('Case 1: no correlation')

    plt.subplot(232)
    plt.plot(case2, alpha=.8);
    plt.xlabel('time')
    plt.ylabel('returns')
    plt.title('Case 2: 0.5 correlation')

    plt.subplot(233)
    plt.plot(case3, alpha=.8);
    plt.xlabel('time')
    plt.ylabel('returns')
    plt.title('Case 3: perfect positive correlation')

    plt.subplot(234)
    plt.plot(case4, alpha=.8);
    plt.xlabel('time')
    plt.ylabel('returns')
    plt.title('Case 4: -0.5 correlation')

    plt.subplot(235)
    plt.plot(case5, alpha=.8);
    plt.xlabel('time')
    plt.ylabel('returns')
    plt.title('Case 5: perfect negative correlation')

    fig.tight_layout()
    return

def statistics(weights, rets_mean, sigma):
    
    weights = np.array(weights) 
    pret = np.sum(rets_mean * weights)
    pvol = np.dot(weights.T, np.dot(sigma, weights))
    
    return np.array([pret, pvol, pret / pvol])

def min_func_sharpe(weights):    
    return -statistics(weights)[2]

def min_func_variance(weights):    
    return statistics(weights)[1]

def min_func_port(w):
    return statistics(w)[1]

def extreme_weights(noa):
    
    ew = rand_weights(noa)*np.random.randint(2, size=(noa))
    np.seterr(divide='ignore', invalid='ignore')
    return ew/sum(ew)

def statistics(weights):
    
    weights = np.array(weights) 
    pret = np.sum(rets_mean * weights)
    pvol = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
    
    return np.array([pret, pvol, pret / pvol])

def min_func_port(w):
    return statistics(w)[1]

def port_opt(rets_mean, sigma, noa, Rf):
   
    weights = np.zeros((2000,noa))
    for i in range(len(weights)):
        weights[i] = rand_weights(noa)
    
    ex_weights = np.zeros((10000,noa))
    for i in range(len(ex_weights)):
        ex_weights[i] = extreme_weights(noa)
    
    new_weights = np.concatenate((weights, ex_weights, np.identity(noa)))

    pret= portf_mean(new_weights, rets_mean)
    pvol = np.sqrt(portf_var(new_weights, sigma))
    
    trets = np.linspace(0, max(pret))
    tvols = []
    
    for tret in trets:
        bnds = tuple((0, 1) for x in range(noa))
        cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret}, 
                {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
        res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',  
                           bounds=bnds, constraints=cons)
        tvols.append(res['fun'])
       
    tvols = np.array(tvols)
    
    ind = np.argmin(tvols)
    evols = tvols[ind:]
    erets = trets[ind:]

    tck = sci.splrep(evols, erets)
    
    Sr = []
    for sig, mu in zip(evols, sci.splev(evols, tck, der = 0)):
         Sr.append((mu - Rf)/sig)
        
    sigma_T = np.linspace(0.0, max(evols))
    E_T = Rf + max(Sr)*sigma_T

    TP = np.argmax(Sr)
    
    con = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - erets[TP]}, 
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    solve = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',bounds=bnds, constraints=con)
    
    keys = rets_mean.index
    values = solve['x'].round(6)
    opt = {keys[i]: values[i] for i in range(len(keys))} 

    plt.figure(figsize=(22,9))

    # random portfolio composition
    plt.scatter(pvol, pret, marker= 'o', lw = 0.1, c = pret/pvol, edgecolors='w')

    # efficient frontier
    plt.plot(evols, erets, 'deeppink', lw=4, label='Efficient frontier')

    # capital market line
    plt.plot(sigma_T, E_T, lw = 3, label= 'Capital Market Line')

    # tangency portfolio
    plt.plot(evols[TP], erets[TP], 'y*', markersize=25.0, label='Tangency portfolio')
    
    plt.grid(True)
    plt.axhline(0, color='k', ls='--', lw=2.0)
    plt.axvline(0, color='k', ls='--', lw=2.0)
    plt.xlabel('expected volatility', fontsize = 15)
    plt.ylabel('expected return', fontsize = 15)
    plt.colorbar(label='Sharpe ratio')
    plt.legend(fontsize=12, loc = 'best')
    plt.title('Capital market line and tangency portfolio', fontsize = 15)
    
    exp_rets = erets[TP]
    exp_vol = evols[TP]

    
    return opt, exp_rets, exp_vol