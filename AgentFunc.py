from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import pandas as pd
import numpy as np


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
            
                #     return sum(prosumer.x[i,t+k] for k in range(0,prosumer.d[i])\
                #                for t in prosumer.Ts)\
                # >= prosumer.d[i]*prosumer.y[i,t]
               
               
               #     return sum(prosumer.x[i,t+k] for k in range(0,prosumer.d[i]))\
               # >= prosumer.d[i]*prosumer.y[i,t]
    #        else: 
    #            return Constraint.Skip 
        
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



def Appliances(n):
    'Randomly generates a set of n appliances'
    'BUG: The biggest d must comes first since in Agent_C set Ts is defined' 
    'as a function of the first element'
    
    #     n=1
    # p=[4,4,4,3,4,2,1,2,4,3,5,6,7,5]
    # p=[0.4*k for k in p]; 
    p=10*[1.6,1.6,1.6,1.2,1.6,0.8,0.4,0.8,1.6,1.2,2.0,2.4,2.8,2.0]
    d=10*[12,12,8,6,5,4,3,4,2,3,2,6,7,4]
    
    # p=p*n
    
    # d=10*[4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    # d=10*[12,6,6,6,6,6,6,6,6,6,6,6,6,6]
    
    # d=d*n

    # p=10*[4,4,4,3,5,2,1,2,5,3,5,6,7,5]
    # p=[0.4*k for k in p]; 
    # d=10*[12,12,8,6,5,4,3,4,2,3,2,6,7,4]
    
    # p=10*[4,4,4,4,4,4,4,4,4,4,4,4,4,4]
    # p=[0.4*k for k in p]; 
    # d=10*[2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    
    # p=[2.8,2.8,2.4,2.4,2.0,2.0,2.0,2.0,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.6,1.2,1.2, \
    #    1.2,1.2,0.8,0.8,0.8,0.8,0.4,0.4]

    # d=[12,7,6,6,2,4,2,4,12,8,12,8,5,5,12,2,12,12,12,2,3,3,6,6,4,4,4,4,3,3]

    
    df=pd.DataFrame({'Power': p, 'Duration': d})
    # print(df)
    df_shuffle=df.sample(n=n)
    # print(df_shuffle)
    # df_shuffle=df.iloc[np.random.permutation(n)]

    df_shuffle.reset_index(drop=True)
    
    p_out=df_shuffle['Power']
    d_out=df_shuffle['Duration']
    
    
    
    # return list(p_out.astype(np.float64)), list(d_out)
    return list(p_out), list(d_out)
 
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

