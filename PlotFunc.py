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
