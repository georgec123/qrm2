import pandas as pd
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
    data[['normal_var_0.95','normal_es_0.95','normal_var_0.99','normal_es_0.99']].plot(ax=ax, style=['r--','r-','b--','b-'], linewidth=0.5)


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
    for idx, var in enumerate(['0.95', '0.99']):
        ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6));
        ax = data[[f'normal_var_{var}']].plot(ax=ax, style=['b--'], linewidth=0.5)

        viols = data[data['max_loss']>data[f'normal_var_{var}']]
        ax.scatter(viols.index,viols['max_loss'], marker='o', c='r', s=10, zorder=10)
        ax.set_title(f"{title}. VaR {var}% violations")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")

        plt.show()



    num_days = (~data['normal_var_0.95'].isna()).sum()
    viols_95 = (data['loss']>data['normal_var_0.95']).sum()
    viols_99 = (data['loss']>data['normal_var_0.99']).sum()

    print(f"Violations 95%: {viols_95}, {100*viols_95/num_days:.2f}%")
    print(f"Violations 99%: {viols_99}, {100*viols_99/num_days:.2f}%")

def print_viols_and_plot_GPD(df: pd.DataFrame, title: str):
    """
    Print summary of data including: 99 VaR violations, 95 VaR violations, and VaR/ES plot
    """
    data = df.copy()
    data.dropna(inplace=True)
    ylabel = "Daily portfolio loss (%) (positive part)"
    
    ################ plot main chart################
    ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6))
    data[['GPD_var_0.95','GPD_es_0.95','GPD_var_0.99','GPD_es_0.99']].plot(ax=ax, style=['r--','r-','b--','b-'], linewidth=0.5)


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
    for idx, var in enumerate(['0.95', '0.99']):
        ax = data[['max_loss']].plot(c='orange', linewidth=0.5, figsize=(10, 6));
        ax = data[[f'GPD_var_{var}']].plot(ax=ax, style=['b--'], linewidth=0.5)

        viols = data[data['max_loss']>data[f'GPD_var_{var}']]
        ax.scatter(viols.index,viols['max_loss'], marker='o', c='r', s=10, zorder=10)
        ax.set_title(f"{title}. VaR {var}% violations")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")

        plt.show()



    num_days = (~data['GPD_var_0.95'].isna()).sum()
    viols_95 = (data['loss']>data['GPD_var_0.95']).sum()
    viols_99 = (data['loss']>data['GPD_var_0.99']).sum()

    print(f"Violations 95%: {viols_95}, {100*viols_95/num_days:.2f}%")
    print(f"Violations 99%: {viols_99}, {100*viols_99/num_days:.2f}%")
