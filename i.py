import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import scipy.stats as ss
import statsmodels.api as sm

def get_q2_data():
    df = pd.read_csv('QRM-2022-cw2-data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True)
    df.set_index('Date')
    df['logreturn'] = np.log(1 + df['TSLA'].pct_change())
    df['loss'] = -100 * df['logreturn']
    df['max_loss'] = df['loss'].apply(lambda x: np.max([x,0]))
    return df.set_index('Date')

def get_training_test_data(dt):
    df = get_q2_data()
    df_training = df[:dt]
    df_test = df[dt:]

    return [df_training, df_test]

x = get_training_test_data('11/25/2021')
train = x[0]
test = x[1]
model = arch_model(train['loss'].dropna(),
                    mean='constant', 
                    vol='GARCH', 
                    p=1, q=1, rescale=True, dist='normal')

model_fit = model.fit(update_freq=-1, disp=0)
print(model_fit.summary())
train['Standardised residuals'] = model_fit.std_resid

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

def plot_acf_graphs(x: pd.Series, title: str):
    fig, axs = plt.subplots(2,1)
    ax1, ax2 = axs

    ax1.set_title(title)
    ax1.set_ylabel(title)
    ax1.plot(x, c='b')

    ax2.set_ylabel('ACF')
    ax2.set_xlabel('Time')

    sm.graphics.tsa.plot_acf(x, ax=ax2, lags=50, title=None, c='orange')

qqplot(train['Standardised residuals'].dropna())
plt.show()
plot_acf_graphs(train['Standardised residuals'].dropna(), 'Standardised residuals (%)')
plt.savefig('plots/Q1ia.png')
plot_acf_graphs(np.abs(train['Standardised residuals'].dropna()), '|Standardised residuals|')
plt.savefig('plots/Q1ib.png')
plot_acf_graphs(train['Standardised residuals'].dropna()**2, 'Squared Standardised residuals')
plt.savefig('plots/Q1ic.png')
plt.show()