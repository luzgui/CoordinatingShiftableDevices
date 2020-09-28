from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np
import random


## SINGLE AGENT PROBLEM ##

def Agent(H,d,p,c,miu):   

    prosumer = ConcreteModel()
    # SETS
    prosumer.T = RangeSet(0,H-1)
    prosumer.Ts = RangeSet(0,H-d-1)

    prosumer.c=Param(prosumer.T, initialize=c)
    
    prosumer.p=Param(initialize=p)
    
    prosumer.d=Param(initialize=d)
    
    # VARIABLES

    # Starting variable
    prosumer.y = Var(prosumer.T,domain=Binary, initialize=0)
    # Activity variable
    prosumer.x = Var(prosumer.T,domain=Binary, initialize=0)
    
    ## CONSTRAINTS
    def Consty(prosumer,t):
        return sum(prosumer.y[t] for t in prosumer.Ts) == 1
    prosumer.y_constraint = Constraint(prosumer.Ts,rule=Consty)

    
    def Constxy(prosumer,t):
    
        return sum(prosumer.x[t+i]for i in range(0,d)) >= prosumer.d*prosumer.y[t]
    prosumer.xy_constraint = Constraint(prosumer.Ts,rule=Constxy)
    
    def Constx(prosumer,t):
        
        return sum(prosumer.x[t] for t in prosumer.T) == prosumer.d
    prosumer.x_constraint = Constraint(prosumer.T,rule=Constx) 
    
    #OBJECTIVE
    
    def MinCost(prosumer):
        return sum(prosumer.x[t]*prosumer.c[t]*prosumer.p*miu for t in prosumer.T)
    prosumer.objective = Objective(rule=MinCost)
    

    return prosumer


## CENTRALIZED PROBLEM ##

def Agent_C(H,nI,d0,p0,c,miu,Viol,Ppv):  

    prosumer = ConcreteModel()
    # SETS
    prosumer.T = RangeSet(0,H-1)
    
    prosumer.I = RangeSet(0,nI-1)
    
    
    prosumer.c=Param(prosumer.T, initialize=c)
    
    prosumer.p=Param(prosumer.I,initialize=p0)
    
    prosumer.d=Param(prosumer.I,initialize=d0)
    
    prosumer.H=Param(initialize=H)
    
    # prosumer.miu=Param(prosumer.I, initialize=miu)
    
    def BuildTs(model,nI):
        for i in range(nI):
                return range(model.d[i],H-model.d[i])
            
    prosumer.Ts = Set(initialize=BuildTs(prosumer,nI))
    # prosumer.Ts = RangeSet(d0,H-d0)
            
    
    # VARIABLES
    
    # Starting variable
    prosumer.y = Var(prosumer.I,prosumer.T,domain=Binary, initialize=0)
    # Activity variable
    prosumer.x = Var(prosumer.I,prosumer.T,domain=Binary, initialize=0)
    
    prosumer.P = Var(prosumer.I,prosumer.T,domain=PositiveReals, initialize=1)
    
    ## CONSTRAINTS
    def Consty(prosumer,i,t):
        return sum(prosumer.y[i,t] for t in prosumer.Ts) == 1
    prosumer.y_constraint = Constraint(prosumer.I,prosumer.Ts,rule=Consty)
    
    #
    def Constxy(prosumer,i,t):
        # for t in prosumer.Ts:
    #        if t >= prosumer.d[i] or t <= H-prosumer.d[i]:
                return sum(prosumer.x[i,t+k] for k in range(0,prosumer.d[i]))\
            >=prosumer.d[i]*prosumer.y[i,t]
        
    prosumer.xy_constraint = Constraint(prosumer.I,prosumer.Ts,rule=Constxy)
    
    
    def ConstP(prosumer,i,t):
        return prosumer.P[i,t] == prosumer.x[i,t]*prosumer.p[i]
    prosumer.PConstraint = Constraint(prosumer.I,prosumer.T, rule=ConstP) 
    
    
    def ConstTotal(prosumer,t):
        return sum(prosumer.x[i,t]*prosumer.p[i] for i in prosumer.I) <= Ppv[t,0]+Viol[t]
    prosumer.TotalConstraint = Constraint(prosumer.T, rule=ConstTotal) 
    
    
    def Constx(prosumer,i,t):    
        return sum(prosumer.x[i,t] for t in prosumer.T) == prosumer.d[i]
    prosumer.x_constraint = Constraint(prosumer.I,prosumer.T,rule=Constx) 
    
    
    #OBJECTIVE
    def MinCost(prosumer):
        return sum(sum(prosumer.x[i,t]*prosumer.c[t]*prosumer.p[i]*miu for t in prosumer.T)\
                   for i in prosumer.I) 
    prosumer.objective = Objective(rule=MinCost)


    return prosumer



def Appliances(N_max, n_max, p_max, d_max, AppsFolder):
    'Randomly generates a set of files with a set of n appliances each file'
    'BUG: The biggest d must comes first since in Agent_C set Ts is defined' 
    'as a function of the first element'
    
    'N_max: Number of sets of appliances to be generated (number of files)'

    for k in range(N_max):
        p=[]
        d=[]
        for i in range(n_max):
            p.append(round(random.uniform(1, p_max),1))
            d.append(round(random.uniform(1, d_max)))
    
        df=pd.DataFrame({'Power': p, 'Duration': d})
        df.reset_index(drop=True)
            
        p_out=df['Power']
        d_out=df['Duration']
        FileName='AppsList_%i.csv' %k
        print(FileName)
        df.to_csv(AppsFolder+'/'+FileName)
    
    # return list(p_out), list(d_out)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

