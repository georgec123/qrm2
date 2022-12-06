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
from preliminary import train, test
from pyextremes import plot_mean_residual_life, plot_parameter_stability, plot_threshold_stability
from pyextremes import EVA
from pyextremes import get_extremes
from pyextremes.plotting import plot_extremes
model = EVA(data=train['Standardised residuals'].dropna())
model.get_extremes("POT", threshold=1, r="24H")
model.plot_extremes(show_clusters=True)

#plot_mean_residual_life(train['Standardised residuals'].dropna())
#plot_parameter_stability(train['Standardised residuals'].dropna())
#plot_threshold_stability(train['Standardised residuals'].dropna(), return_period=100, thresholds=np.linspace(1.2, 1.8, 20))
plt.show()