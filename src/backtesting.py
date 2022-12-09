import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats.distributions import chi2
import matplotlib.dates as mdates

from typing import Tuple



def LL_GPD(xi: float, beta: float, ys: np.ndarray) -> float:
    # Part 4, slide 43
    N = len(ys)

    first = -N*np.log(beta)
    second = (1+1/xi)*np.sum(np.log(1+ys*xi/beta))
    return first - second

def GPD_cdf(x: float, xi: float, beta: float) -> float: 
    # Implement the CDF of GPD
    # Part 4, slide 38
    if xi == 0:
        return 1 - np.exp(-x/beta)
    else:
        return 1 - (1 + xi * x / beta) ** (-1/xi)

def empirical_F(dist: np.ndarray, x: float):
    # assuming dist has a CDF of F, we return P(X<x) = F(x)
    lt_x = dist[dist<x]
    return len(lt_x)/len(dist)

def q_alpha_pareto(zs: np.ndarray, alpha: float, xi: float, beta: float, u: float) -> np.ndarray:
    # part 4, slide 62
    q_alpha = u +  (beta/xi)*( ((1-alpha)/(1-empirical_F(zs, u)))**(-1*xi)  -1)

    return q_alpha

def es_pareto_mult(q: float, beta: float, xi: float, u: float) -> float:
    # part 4, slide 62

    return (q + beta-(xi*u))/(1-xi)


def fit_gpd(dist: pd.Series, u: float ) -> Tuple[float, float]:
    """
    Fit a Generalised Pareto distribution

    :param dist: Distrubution to fit GPD to
    :param u: u to create excess distribution

    :returns: xi, beta

    """
    excess_dist = dist[dist>u] - u

    bnds = ((0.1,30), (0,30))

    res = minimize(fun = lambda theta: -1*LL_GPD(theta[0], theta[1], ys=excess_dist),
        x0 = (10,10),
        method='SLSQP',
        bounds=bnds,
        options={'disp': False})

    xi, beta = res.x
    return xi, beta

def compute_var_df(df: pd.DataFrame, q_95: float, q_99: float, es_95_mult: float, es_99_mult: float) -> pd.DataFrame:
    
    data = df.copy()
    data['var_95'] = data['mu'] + data['std_forc']*q_95
    data['var_99'] = data['mu'] + data['std_forc']*q_99

    data['es_95'] = data['mu'] + data['std_forc']*es_95_mult
    data['es_99'] = data['mu'] + data['std_forc']*es_99_mult
    
    return data


def mean_excess(x: np.ndarray, u: float):
    # ref Part 4 - slide 26
    exceedances = x[x>u]
    std_exceedances = exceedances-u
    return np.mean(std_exceedances)


def print_viols_and_plot(df: pd.DataFrame, title: str, q: str):
    """
    Print summary of data including: 99 VaR violations, 95 VaR violations, and VaR/ES plot
    """
    data = df.copy()
    data.dropna(inplace=True)
    ylabel = "Daily portfolio loss (%) (positive part)"
    
    ################ plot main chart################
    ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6))
    data[['var_95','es_95','var_99','es_99']].plot(ax=ax, style=['r--','r-','b--','b-'], linewidth=0.5)


    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")

    months = mdates.MonthLocator((1,4,7,10))
    ax.xaxis.set_minor_locator(months)
    
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    
    ax.set_title(title)
    ax.legend(loc='upper left')
    plt.savefig(f'../plots/iv/iv_{q}i.png')

    plt.show()

    ################ plot var viols ################
    for idx, var in enumerate(['95', '99']):
        ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6));
        ax = data[[f'var_{var}']].plot(ax=ax, style=['b--'], linewidth=0.5)

        viols = data[data['max_loss']>data[f'var_{var}']]
        ax.scatter(viols.index,viols['max_loss'], marker='o', c='r', s=10, zorder=10)
        ax.set_title(f"{title}. VaR {var}% violations")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        plt.savefig(f'../plots/iv/iv_{q}{"i"*(idx+2)}.png')

        plt.show()



    num_days = (~data['var_95'].isna()).sum()
    viols_95 = (data['loss']>data['var_95']).sum()
    viols_99 = (data['loss']>data['var_99']).sum()

    print(f"Violations 95%: {viols_95}, {100*viols_95/num_days:.2f}%")
    print(f"Violations 99%: {viols_99}, {100*viols_99/num_days:.2f}%")


def str_ci(x):
    return str(x).split('.')[1]

def log_lik(p: float, obs: pd.Series) -> float:
    """
    Likelihood ratio = -2*log(L_1/L_2)
    This function will calculate log(L_i) for any series of binomial obesrvations

    :param p: probability of success
    :param obs: Series of observations, values are 1 or 0
    """
    return obs.apply(lambda x: np.log(((1-p)**(1-x))*(p**x))).sum()

def lr_uc(alpha: float, exceeds: pd.Series) -> float:
    """
    Calculate unconditional likelihood ratio statistic for any series of binomial obesrvations

    :param alpha: confidence level between 0 and 1
    :param exceeds: Series of booleans/ints, indicating if the event happened
    """
    # calc pi_hat MLE 
    obs = exceeds.astype(int)
    pi_hat = np.mean(obs)

    # return LR_UC for pi_hat and 1-alpha
    return -2*(log_lik(1-alpha, obs)-log_lik(pi_hat, obs))

def p_chi2(x: float, dof: int=1) -> float:
    """
    Calculate the p value for a chi2 distrubution

    :param x: test statistic value
    :param dof: Degrees of freedom for chi2 ist
    """
    return 1-chi2.cdf(x, dof)


def get_stats(data: pd.DataFrame, title: str) -> pd.DataFrame: 
    
    df=pd.DataFrame(columns=['Alpha', 'Violations (exp)', 'LR_uc', 'LR_uc - p'])

    for alpha in [0.95, 0.99]:

        # str alpha for dataframe column lookup
        str_alpha = str_ci(alpha)

        # remove start data where there is no var
        non_na_data = data[~data[f'var_{str_alpha}'].isna()]

        num_days = len(non_na_data)
        viol_mask = (non_na_data['loss']>non_na_data[f'var_{str_alpha}'])
        num_viols = viol_mask.sum()


        # calculate p values for var
        expected_viols = (1-alpha)*num_days

        likelihood_uc = lr_uc(alpha, viol_mask)
        p_val_uc = p_chi2(likelihood_uc)

        df.loc[len(df),:] = [
                        alpha,
                        f"{num_viols} ({expected_viols:.1f})",
                        f"{likelihood_uc:.3f}",
                        f"{p_val_uc:.5f}"
                        ]
                        
    df.insert(0, 'Title', title)
    return df