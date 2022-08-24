#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 07:20:19 2021

stats presentation

@author: sean
"""

import numpy as np
import matplotlib.pyplot as plt

mu1, sigma1 = 50, 25
s1 = np.random.normal(mu1, sigma1, 1000)

mu2, sigma2 = 51, 24
s1 = np.random.normal(mu1, sigma1, 1000)

st_err10 = sigma1/np.sqrt(10)
st_err50 = sigma1/np.sqrt(50)

plt.figure(); plt.hist(s1)

means = np.mean(s1[0:9]), np.mean(s1[0:9])
err = sigma1, st_err10
fig, ax = plt.subplots()
ax.bar(means, yerr = err)

sample1 = s1[0:20]
sample2 = s1[21:30]
sample3 = s1[30:35]

plt.figure()
plt.hist(sample1, label = 'sample 1', alpha = 0.5, density = True)
plt.hist(sample2, label = 'sample 2', alpha = 0.5, density = True)
plt.hist(sample3, label = 'sample 3', alpha = 0.5, density = True)
plt.hist(s1, label = 'True population', alpha = 0.5, density = True)
plt.legend(loc = 'upper right')