import matplotlib.pyplot as plt 
import statsmodels.api as sm
import pandas as pd
import numpy as np
import scipy.stats as ss
from scipy.stats.distributions import norm

from backtesting import GPD_cdf

def qqplot(x, dist="norm"):
    fig, ax = plt.subplots(1,1)
    t, o = ss.probplot(x, dist=dist, plot=ax)
    
    if isinstance(dist, str):
        title = f"{dist.title()} Q-Q Plot"
    else:
        title = f"{dist.dist.name.title()} Q-Q Plot"
        
    
    min_ = min(t[0])
    max_ = max(t[0])
    
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(title)

    ax.scatter(t[0],t[1])
    ax.plot((min_, min_), (max_,max_))

    return ax

def get_linspace(data, num_stdevs=5, points=1001):
    """
    Return linspace of the data by calculating (mu-num_stdevs*std, mu+num_stdevs*std)
    """
    mean = np.mean(data)
    stdev = np.std(data)

    linspace = np.linspace(mean - num_stdevs*stdev, mean + num_stdevs*stdev, points)

    return linspace


def plot_acf_graphs(x: pd.Series, title: str):
    fig, axs = plt.subplots(2,1)
    ax1, ax2 = axs

    ax1.set_title(title)
    ax1.set_ylabel(title)
    ax1.plot(x, c='b')

    ax2.set_ylabel('ACF')
    ax2.set_xlabel('Lag')

    sm.graphics.tsa.plot_acf(x, ax=ax2, lags=50, title=None, auto_ylims=True, c='orange')


def plot_excess_dist(y, xi, beta, u):
    xx = np.linspace(0, 250, 10000) # For plotting the theoretical, fitted GPD

    # Empirical CDF functionality from the "statsmodels" package
    empirical_cdf_obj = sm.distributions.empirical_distribution.ECDF((y.to_numpy()).flatten()) 

    fig, ax = plt.subplots()
    ax.plot(xx, GPD_cdf(xx, xi, beta), 'k-', label='Fitted') # Plot the CDF of fitted GPD
    ax.plot(y, empirical_cdf_obj(y), 'b*', label='Empirical') # Plot the empirical CDF of Y, the excess claim value above u
    ax.legend()
    ax.set_xscale('log')
    ax.set_title(f'Empirical vs fitted CDF of the excess distribution, u={u}')
    ax.set_ylabel('$F_{u}(x-u)$')
    ax.set_xlabel('$x$')
    return ax

def histogram_w_kde_and_dist(x, title: str, dist=None, dist_name='Normal'):
    if dist is None:
        dist = norm()

    linspace = get_linspace(x)
    kde = ss.gaussian_kde(x)
    dist_fit_pdf = dist.pdf(linspace)

    fig, ax = plt.subplots()
    ax.plot(linspace, dist_fit_pdf, label= f'{dist_name} Distribution Fit', color='red');
    ax.plot(linspace, kde(linspace), label='KDE', color='k');
    ax.hist(x, bins=50, density=True,facecolor='#2ab0ff', edgecolor='white');
    ax.set_ylabel('Density')
    ax.set_title(title)
    plt.legend()
    
    return ax
