import numpy as np
from arch import arch_model
import statsmodels.api as sm
from preliminary import train, test
from iiib import sol, u
from plotting import print_viols_and_plot_GPD, print_viols_and_plot_normal
import statsmodels.api as sm

def es(loss: np.ndarray, var: np.ndarray):
    """
    Calculate expected shortfall.
    Takes array of length n+1, and calculates avg(loss) of first n items given loss[0:n]>var[-1]
    :param loss: array of n+1 observations
    :param var: array of n+1 observations, but we only care about the 'final' element 
    """
    # check if loss[0 to n inclusive] > var[n+1]
    loss = loss[:-1]
    breach_mask = loss>var[-1]
    if not breach_mask.sum():
        return 0
    return loss[breach_mask].sum() / breach_mask.sum()

def es_n(loss: np.ndarray, var: float):
    """
    Calculate expected shortfall.
    Takes array of length n, and calculates avg(loss) given loss[0:n]>var
    :param loss: array of n observations, previous loss 
    :param var: Value at Risk
    """
    # check if loss[0 to n inclusive] > var
    breach_mask = loss>var
    if not breach_mask.sum():
        # return = for no breaches
        return 0
    return loss[breach_mask].sum() / breach_mask.sum()

model = arch_model(train['loss'].dropna(),
                    mean='constant', 
                    vol='GARCH', 
                    p=1, q=1, rescale=True, dist='normal')

model_fit = model.fit(update_freq=-1, disp=0)
forecasts = model_fit.forecast(horizon=len(test), reindex=False)
test['normal_mean'] = forecasts.mean.T.values
test['normal_variance'] = forecasts.variance.T.values
ecdf = sm.distributions.ECDF(model_fit.std_resid)
print(test)

def VaR_ES_normal(df, model_fit, alpha):
    q = np.quantile(model_fit.std_resid.values, alpha)
    std = np.sqrt(df['normal_variance'])
    value_at_risk = df['normal_mean'] + q * std
    df['normal_var_' + str(alpha)] = value_at_risk

    res_exp_sh = es_n(model_fit.std_resid, q)
    exp_sh = df['normal_mean'] + (df['normal_variance']**0.5)*res_exp_sh
    df['normal_es_' + str(alpha)] = exp_sh

    return df

VaR_ES_normal(test, model_fit, 0.95)
VaR_ES_normal(test, model_fit, 0.99)
print(test)
print_viols_and_plot_normal(test, 'A')

def VaR_ES_GPD(df, alpha, u, xi, beta):
    q = u + ((beta/xi) * ((((1-alpha)/(1-ecdf(u)))**-xi) - 1))
    value_at_risk = df['normal_mean'] + (df['normal_variance']**0.5)*q
    df['GPD_var_' + str(alpha)] = value_at_risk

    p = (q + beta-(xi*u))/(1-xi)
    es = df['normal_mean'] + (df['normal_variance']**0.5)*p
    df['GPD_es_' + str(alpha)] = es

    return df

xi = sol.x[0]
beta = sol.x[1]
print(xi)
print(beta)
VaR_ES_GPD(test, 0.95, u, xi, beta)
VaR_ES_GPD(test, 0.99, u, xi, beta)
print(test)
print_viols_and_plot_GPD(test, 'B')