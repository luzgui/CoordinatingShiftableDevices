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
    
    #make figure
    fig, ax1 = plt.subplots()
    fw=14
    
    color = 'tab:gray'
    color2='tab:orange'
    ax1.set_xlabel('hour of the day', fontsize=fw)
    ax1.set_ylabel('€/kWh', color=color,fontsize=fw)
    ax1.plot(T,np.asarray(list(c)), color=color,linestyle='dashed')
    ax1.tick_params(axis='y',labelsize=fw)
    ax1.set_title('CP N=%i' %n)
    div=12
    L=np.linspace(0,H,div,endpoint=False)
    ax1.set_xticks(L)
    ax1.set_xticklabels(np.linspace(0,24,div,dtype=int,endpoint=False),fontsize=fw)

    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis'
    ax2.set_ylabel('kW', color=color2)  # we already handled the x-label with ax1
    # ax2.plot(T,power)
    
    for k in range(len(P_Raw)):
       ax2.plot(T,P_Raw[k,:]) 
        
    ax2.plot(T, Pag, color='black',linewidth=3.0) 
    # ax2.tick_params(axis='y', labelcolor=color)

    # ax2.set_xlabel('hour of the day')
    ax2.set_ylabel('kW', color=color2,fontsize=fw)
    ax2.plot(T,Ppv, color='tab:orange')
    ax2.tick_params(axis='y', labelsize=fw)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    file=ResultsFolder + '/CP_N_%i' %n
    plt.savefig(file,dpi=300)
    plt.show()

    
    
    
    
    
    
    
    # color = 'tab:red'
    # ax1.set_xlabel('hour of the day',fontsize=fw)
    # ax1.set_ylabel('euro/kWh', color=color,fontsize=fw)
    # ax1.plot(T,np.asarray(list(c)), color=color)
    # ax1.tick_params(axis='y', labelcolor=color,labelsize=fw)
    
    
    # ax1.set_title('CP N=%i' %n)
    
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    # color = 'tab:blue'
    # color2= 'tab:green'
    # ax2.set_ylabel('kW', color=color)  # we already handled the x-label with ax1
    
    # for k in range(len(P_Raw)):
    #    ax2.plot(T,P_Raw[k,:]) 
    
    # ax2.plot(T,Pag , color='black',linewidth=3.0)
    # ax2.tick_params(axis='y', labelcolor=color)
    
    # color = 'tab:red'
    # ax2.set_xlabel('hour of the day')
    # ax2.set_ylabel('kW', color=color2)
    # ax2.plot(T,Ppv, color=color2)
    # ax2.tick_params(axis='y', labelcolor=color)
    
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # file=ResultsFolder + '/CP_N_%i' %n
    # plt.savefig(file,dpi=300)
    
    # plt.show()


def DevScat(p,d,ResultsFolder,n):
    'Gerate a scatter for the generated devices with points (p,d)=(power, duration)'
    fw=14
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
    plt.xlabel("power demand (kW)", fontsize=fw)
    plt.ylabel("Duration (nº timeslots)", fontsize=fw)
    plt.colorbar()
    plt.grid(True)
    plt.title('Shiftable devices distribution N=%i' %n, fontsize=fw)
    file=ResultsFolder + '/Scatter_N_%i' %n
    plt.savefig(file,dpi=300)
    plt.show()


def PlotCompare(df,N,ResultsFolder):
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
    
    file=ResultsFolder + '/Compare'
    plt.savefig(file,dpi=300)
    plt.show()