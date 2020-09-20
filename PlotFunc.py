from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import os



def PlotFunc_Central(Model,Ppv, n, ResultsFolder):
    "Model argument is a Pyomo model with results, n: number of agents, Ppv: PV capacity,"
    "Resultsfolder: destination of plots"

    #importing values
    P=array([value(Model.P[i,t]) for i in Model.I for t in Model.T])
    x=array([value(Model.x[i,t]) for i in Model.I for t in Model.T])
    c=array([value(Model.c[t]) for t in Model.T])
    H=len(c)
    nI=len(Model.I)
    
    xx=np.reshape(x,(nI,H))
    PP=np.reshape(P,(nI,H))
    
    P_Raw=np.empty(shape=(nI,H)); P_Raw.fill(0)
    for i in Model.I:
        P_Raw[i,:]=[value(Model.x[i,t])*value(Model.p[i]) for t in Model.T]
    
    Pag=P_Raw.sum(axis=0)
    T=array(list(Model.T))
    
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('hour of the day')
    ax1.set_ylabel('euro/kWh', color=color)
    ax1.plot(T,np.asarray(list(c)), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    
    ax1.set_title('CP Nagents: %i' %n)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    color2= 'tab:green'
    ax2.set_ylabel('kW', color=color)  # we already handled the x-label with ax1
    
    for k in range(len(P_Raw)):
       ax2.plot(T,P_Raw[k,:]) 
    
    ax2.plot(T,Pag , color='black',linewidth=3.0)
    ax2.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:red'
    ax2.set_xlabel('hour of the day')
    ax2.set_ylabel('kW', color=color2)
    ax2.plot(T,Ppv, color=color2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    file=ResultsFolder + '/CP_N_%i' %n
    plt.savefig(file,dpi=200)
    
    plt.show()


def DevScat(p,d):
    fig = plt.figure()
    #Generate a list of unique points
    points=list(set(zip(p,d))) 
    #Generate a list of point counts
    count=[len([x for x,y in zip(p,d) if x==pp[0] and y==pp[1]]) for pp in points]
    #Now for the plotting:
    plot_x=[i[0] for i in points]
    plot_y=[i[1] for i in points]
    count=np.array(count)
    plt.scatter(plot_x,plot_y,c=count,s=100*count**0.5,cmap='Spectral_r')
    plt.colorbar()
    plt.grid(True)
    plt.show()


def PlotCompare(df):
    'df is a data frame'
    df_DP=pd.DataFrame
    df_DP=df[df.Model.str.contains('DP', case=True)]
    df_DP=df_DP.sort_values(by='Model')
    df_DP=df_DP.reset_index(drop=True)

    df_CP=pd.DataFrame
    df_CP=df[df.Model.str.contains('CP', case=True)]
    df_CP=df_CP.sort_values(by='Model')
    df_CP=df_CP.reset_index(drop=True)
    t.astype(float).plot.bar()
    
    
    N=[15,25,35,45,55,65,75,85]
    
    
    fig, axs = plt.subplots(2, 2)

    axs[0,0].plot(N,df_DP['Wall_Time'].astype(float),color='red')
    axs[0,0].axes.set_xticks(N)
    axs[0,0].legend(['Distributed Problem'])
    axs[0,0].grid()
    axs[0,0].axes.set_xlabel('Number of Agents')
    axs[0,0].axes.set_ylabel('Time(s)')
    axs[0,0].set_title('Wall Time')

    axs[1,0].plot(N,df_CP['Wall_Time'].astype(float))
    axs[1,0].axes.set_xticks(N)
    axs[1,0].legend(['Centralized Problem'],loc='upper left')
    axs[1,0].grid()
    axs[1,0].axes.set_xlabel('Number of Agents')
    axs[1,0].axes.set_ylabel('Time(s)')
    # axs[1,0].set_title('Centralized problem')

    axs[0,1].plot(N,df_DP['Objective'].astype(float),color='red')
    axs[0,1].plot(N,df_CP['Objective'].astype(float))
    axs[0,1].axes.set_xticks(N)
    axs[0,1].grid()
    axs[0,1].axes.set_xlabel('Number of Agents')
    axs[0,1].axes.set_ylabel('€')
    axs[0,1].set_title('Objective function')

    axs[1,1].plot(N,df_DP['SSR'].astype(float),color='red')
    axs[1,1].plot(N,df_CP['SSR'].astype(float))
    axs[1,1].axes.set_xticks(N)
    axs[1,1].grid()
    axs[1,1].axes.set_xlabel('Number of Agents')
    axs[1,1].axes.set_ylabel('%')
    axs[1,1].set_title('Self-Suficiency Ratio')
