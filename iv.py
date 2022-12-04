import numpy as np
from arch import arch_model
import pandas as pd
import datetime as dt
from preliminary import train, test
from plotting import print_viols_and_plot_normal
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def print_viols_and_plot_normal(df: pd.DataFrame, title: str):
    """
    Print summary of data including: 99 VaR violations, 95 VaR violations, and VaR/ES plot
    """
    data = df.copy()
    data.dropna(inplace=True)
    ylabel = "Daily portfolio loss (%) (positive part)"
    
    ################ plot main chart################
    ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6))
    data[['VaR_95','ES_95','VaR_99','ES_99']].plot(ax=ax, style=['r--','r-','b--','b-'], linewidth=0.5)


    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")
    
    months = mdates.MonthLocator((1,4,7,10))
    ax.xaxis.set_minor_locator(months)
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.show()

    ################ plot var viols ################
    for idx, var in enumerate(['95', '99']):
        ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6))
        ax = data[[f'VaR_{var}']].plot(ax=ax, style=['b--'], linewidth=0.5)

        viols = data[data['max_loss']>data[f'VaR_{var}']]
        ax.scatter(viols.index,viols['max_loss'], marker='o', c='r', s=10, zorder=10)
        ax.set_title(f"{title}. VaR {var}% violations")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")

        plt.show()



    num_days = (~data['VaR_95'].isna()).sum()
    viols_95 = (data['loss']>data['VaR_95']).sum()
    viols_99 = (data['loss']>data['VaR_99']).sum()

    print(f"Violations 95%: {viols_95}, {100*viols_95/num_days:.2f}%")
    print(f"Violations 99%: {viols_99}, {100*viols_99/num_days:.2f}%")

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

split = dt.datetime(2021, 11, 26)
def get_q_data():
    df = pd.read_csv('QRM-2022-cw2-data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True)
    df.set_index('Date')
    df['logreturn'] = np.log(1 + df['TSLA'].pct_change())
    df['loss'] = -100 * df['logreturn']
    df['max_loss'] = df['loss'].apply(lambda x: np.max([x,0]))
    return df.set_index('Date')

def get_training_test_data(df, dt):
    df_training = df[:dt]
    df_test = df[dt:]

    return [df_training, df_test]

df = get_q_data()
[train, test] = get_training_test_data(df, split)
model = arch_model(df['loss'].dropna(),
                    mean='constant', 
                    vol='GARCH', 
                    p=1, q=1, rescale=True, dist='normal')

model_fit = model.fit(update_freq=-1, disp=0, last_obs=split)
alpha_0 = model_fit.params['omega']
alpha_1 = model_fit.params['alpha[1]']
beta_1 = model_fit.params['beta[1]']
print(alpha_0)
print(alpha_1)
print(beta_1)
train['Standardised residuals'] = model_fit.std_resid
print(df)
forecasts = model_fit.forecast(horizon=1, start=split ,reindex=False)
test['normal_mean'] = forecasts.mean.T.values[0]
test['normal_variance'] = forecasts.variance.T.values[0]

df['Standardised residuals'] = train['Standardised residuals'].append(pd.Series([0 for x in range(len(test))]))
df['normal_mean'] = pd.Series([0 for x in range(len(train))]).append(test['normal_mean'])
df['normal_variance'] = pd.Series([0 for x in range(len(train))]).append(test['normal_variance'])
#forecasts.variance[split:].plot()
#plt.show()
print(forecasts.variance.T.values[0])
print(df)

VaR_95 = [0 for x in range(len(train))]
VaR_99 = [0 for x in range(len(train))]
ES_95 = [0 for x in range(len(train))]
ES_99 = [0 for x in range(len(train))]

n = 500
for i in range(len(test) - 1):
    st = i + len(train)
    q_95 = np.quantile(df['Standardised residuals'][st - n:st].dropna(), 0.95)
    q_99 = np.quantile(df['Standardised residuals'][st - n:st].dropna(), 0.99)

    variance = alpha_0 + alpha_1 * df['loss'][st]**2 + beta_1*df['normal_variance'][st - 1]
    df['normal_variance'][st] = variance
    new_std_resid = (df['loss'][st] - df['normal_mean'][st]) / df['normal_variance'][st]**0.5
    df['Standardised residuals'][st] = new_std_resid
    VaR_95.append(df['normal_mean'][st] + (df['normal_variance'][st]**0.5 * q_95))
    VaR_99.append(df['normal_mean'][st] + (df['normal_variance'][st]**0.5 * q_99))

    res_exp_sh_95 = es_n(df['Standardised residuals'][st - n:st].dropna(), q_95)
    res_exp_sh_99 = es_n(df['Standardised residuals'][st - n:st].dropna(), q_99)

    exp_sh_95 = df['normal_mean'][st] + (df['normal_variance'][st]**0.5)*res_exp_sh_95
    exp_sh_99 = df['normal_mean'][st] + (df['normal_variance'][st]**0.5)*res_exp_sh_99
    ES_95.append(exp_sh_95)
    ES_99.append(exp_sh_99)

df['VaR_95'] = VaR_95
df['VaR_99'] = VaR_99
df['ES_95'] = ES_95
df['ES_99'] = ES_99
print(df)
print_viols_and_plot_normal(df, 'A')
