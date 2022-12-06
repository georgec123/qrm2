import pandas as pd
import numpy as np
import datetime as dt
from arch import arch_model

def get_q_data():
    df = pd.read_csv('QRM-2022-cw2-data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', ascending=True)
    df.set_index('Date')
    df['logreturn'] = np.log(1 + df['TSLA'].pct_change())
    df['loss'] = -100 * df['logreturn']
    df['max_loss'] = df['loss'].apply(lambda x: np.max([x,0]))
    return df.set_index('Date')

def get_training_test_data(dt):
    df = get_q_data()
    df_training = df[:dt]
    df_test = df[dt:]

    return [df_training, df_test, df]

x = get_training_test_data('11/25/2021')
train = x[0]
test = x[1]
df = x[2]
model = arch_model(train['loss'].dropna(),
                    mean='constant', 
                    vol='GARCH', 
                    p=1, q=1, rescale=True, dist='normal')

model_fit = model.fit(update_freq=-1, disp=0)
mu = model_fit.params['mu']
alpha_0 = model_fit.params['omega']
alpha_1 = model_fit.params['alpha[1]']
beta_1 = model_fit.params['beta[1]']
print(model_fit.summary())
train['Standardised residuals'] = model_fit.std_resid
split = dt.datetime(2021, 11, 26)
print(len(train))
print(len(test))
print(len(train) + len(test))
print(len(df))