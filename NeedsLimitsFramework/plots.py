""" Plotting functions for results from the Needs and Limits Model. """

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def agent_dist(results, x, y, hue=None, t=None,
               xlabel=None, ylabel=None,
               xbins=None, ybins=None, 
               xbinrange=None, ybinrange=None,
               obj_types='Individual', color='tab:blue'):
    
    g = sns.JointGrid(ratio=3, space=.05,)
    df = results.arrange(variables=True, parameters=True, obj_types=obj_types)
    t = t if t else df['t'].max()  # End of simulation
    df = df[df['t'] == t]  # Select single time-step
    hue = hue if hue in df else None
    
    sns.scatterplot(data=df, x=x, y=y, hue=hue, s=30, alpha=.5, ax=g.ax_joint, color=color)
    sns.histplot(data=df, x=x, bins=xbins, binrange=xbinrange, ax=g.ax_marg_x, color=color) 
    sns.histplot(data=df, y=y, bins=ybins, binrange=ybinrange, ax=g.ax_marg_y, color=color) 
    
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    
    
def hide_xticks(ax):
    for ax in make_list(ax):
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1line.set_visible(False)
