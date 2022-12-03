import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize 
import statsmodels.api as sm
from preliminary import train
from pyextremes import EVA
model = EVA(data=train['Standardised residuals'].dropna())
model.get_extremes("POT", threshold=-1, r="24H")
model.plot_extremes(show_clusters=True)
plot.show()

v = np.linspace(-1, 5, 1000) # some values of v
excess_mean = []

for vv in v:
    excess_mean.append(np.sum((train['Standardised residuals'].dropna() - vv) * (train['Standardised residuals'].dropna() > vv)) / np.sum(train['Standardised residuals'].dropna() > vv))

plt.plot(v, excess_mean, 'r*')
plt.title("Sample mean excess function")
plt.ylabel("$e_n(v)$")
plt.xlabel("$v$")
plt.show()


u = 0.8 # our choice of u
y = train[train['loss'] > u]['loss'] - u # construction of y
print(y)

def GPD_loglikelihood(y, xi, beta):
    return -len(y) * np.log(beta) - (1 + 1/xi) * np.sum(np.log(1 + xi * y / beta))

objfun = lambda theta : -1 * GPD_loglikelihood(y, theta[0], theta[1]) # theta[0] represents xi and theta[1] represents beta

theta0 = [1, 1]  # initial guess
print("Initial objective function value: " + str(objfun(theta0)))

bnds = ((0.01,20), (0,20)) # individual bound on each parameter

sol = minimize(objfun,theta0,method='SLSQP',bounds=bnds, options={'disp': True})

print("Estimate of xi: " + str(sol.x[0]))
print("Estimate of beta: " + str(sol.x[1]))

def GPD_cdf(x, xi, beta): # Implement the CDF of GPD
    if xi == 0:
        return 1 - np.exp(-x/beta)
    else:
        return 1 - (1 + xi * x / beta) ** (-1/xi)

xx = np.linspace(0, 5, 100) # For plotting the theoretical, fitted GPD
xxu = np.linspace(u, 5, 100) - u
# Empirical CDF functionality from the "statsmodels" package
empirical_cdf_obj = sm.distributions.empirical_distribution.ECDF((y.to_numpy()).flatten()) 
a = GPD_cdf(u, sol.x[0], sol.x[1])
print(a)

fig, ax = plt.subplots()
ax.plot(xx, GPD_cdf(xx, sol.x[0], sol.x[1]), 'k-', label='Fitted') # Plot the CDF of fitted GPD
ax.plot(y, empirical_cdf_obj(y), 'b*', label='Empirical') # Plot the empirical CDF of Y, the excess claim value above u
ax.legend()
ax.set_xscale('log')
ax.set_title('Empirical vs fitted CDF of the excess distribution')
ax.set_ylabel('$F_{u}(x)$')
ax.set_xlabel('$x$')
plt.show()
