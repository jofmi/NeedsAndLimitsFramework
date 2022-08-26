import sys
sys.path.append("..")
import NeedsLimitsFramework.plots as plots
import agentpy as ap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def qol_income_dist(results, y, ylabel, hue=None, color=None): 
    plots.agent_dist(
        results=results,
        x='income', y=y,  
        xlabel='Income', ylabel=ylabel,
        xbins=30, ybins=11,
        ybinrange=(0,1),
        hue=hue,
        color=color,
        obj_types=('Individual')
    )

    
def growth_redist():
    results_growth = ap.DataDict.load('growth', 0, 'results', display=False)
    results_redist = ap.DataDict.load('redist', 0, 'results', display=False)
    data1 = results_growth.arrange_reporters()
    data2 = results_redist.arrange_reporters()
    data1['growth'] *= 100
    data2['redist'] *= 100

    series = [
        ('Q', 'Quality of life (QOL)', 'tab:purple'),
        ('CES', 'CES Utility', 'tab:red'),
        ('I', 'Income', 'tab:blue'),
    ]
    
    fig, rows = plt.subplots(2, 2, figsize=(6, 6), sharex='col', sharey='row')
    axs = [ax for row in rows for ax in row]
    for X, f_label, c in series:  
        
        x_keys = ['growth', 'redist'] * 2
        y_keys = [f'M({X})'] * 2 + [f'G({X})'] * 2
        datas = [data1, data2] * 2

        for ax, x, y, data in zip(axs, x_keys, y_keys, datas): 
            sns.lineplot(data=data, x=x, y=y, ax=ax, color=c, label=f_label)
        
        for ax in (0,2,3):
            axs[ax].get_legend().remove()
        
    xlabels = ('Income growth [%]', 'Redistribution [%]')
    for ax, label in zip(rows[1], xlabels):
        ax.set_xlabel(label)  # Bottom row

    ylabels = (f'Average - Well-being', f'Gini coefficient - Well-being')
    for row, label in zip(rows, ylabels):
        row[0].set_ylabel(label)  # Left column
    
    plt.tight_layout()
    
    
def carbon_tax():
    results = ap.DataDict.load('ctax', 0, 'results', display=None)
    fig, rows = plt.subplots(2, 2, figsize=(6, 6), sharex='col')
    axs = [ax for row in rows for ax in row]
    hue = 'ctax_redist' 
    y_keys = ('M(E)', 'M(Q)', 'G(E)', 'G(Q)')
    data = results.arrange_reporters()
    for ax, y in zip(axs, y_keys): 
        sns.lineplot(data=data, x='ctax', hue=hue, y=y, ax=ax)
    if hue:
        for ax in axs:
            ax.get_legend().remove()
    plt.tight_layout()
    for row, X in zip(rows, ('Average', 'Gini coefficient')):
        for ax, Y in zip(row, ('Emissions', 'QOL')):
            ax.set_ylabel(f'{X} - {Y}')
    for ax in rows[1]:
        ax.set_xlabel('Carbon tax rate')
    
    
def multiplier():    
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex='col')
    ax1, ax2 = axs
    
    results = ap.DataDict.load('multiplier', 0, 'results', display=False)
    df = results.arrange_reporters()
    hue='imp_norms_recreation'
    x='imp_norms'
    y_keys = ('M(E)', 'M(Q)')
    
    for ax, y in zip(axs, y_keys):
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
    
    ax1.legend(title='Perceived\nimportance\nof recreation\n& rest', 
               bbox_to_anchor=(1,1), loc="upper left")
    ax2.get_legend().remove()
    
    ax1.set_ylabel('Average - Emissions')
    ax2.set_ylabel('Average - QOL')
    ax2.set_xlabel('Perceived importance of social norms')
    
    plt.tight_layout()
