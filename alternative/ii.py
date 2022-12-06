from webbrowser import get
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
import scipy.stats as ss
from scipy.optimize import minimize 
from scipy.stats import t
import statsmodels.api as sm
import math
from preliminary import train, test

t_fitted = t.fit(train['Standardised residuals'].dropna(), floc = 0)
val = t_fitted
nu = val[0]
print(val)
p = t.pdf(train['Standardised residuals'], nu, 0, (math.sqrt((nu - 2)/nu)))
print(p)
