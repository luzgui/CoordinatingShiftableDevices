from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory



def Agent(H,d,p,c):   

    prosumer = ConcreteModel()
    # SETS
    prosumer.T = RangeSet(0,H-1)
    prosumer.Ts = RangeSet(0,H-d-1)

    prosumer.c=Param(prosumer.T, initialize=c)
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
    
        return sum(prosumer.x[t+i]for i in range(0,d)) >= d*prosumer.y[t]
    prosumer.xy_constraint = Constraint(prosumer.Ts,rule=Constxy)
    
    def Constx(prosumer,t):
        
        return sum(prosumer.x[t] for t in prosumer.T) == d
    prosumer.x_constraint = Constraint(prosumer.T,rule=Constx) 
    
#OBJECTIVE
    
    def MinCost(prosumer):
        return sum(prosumer.x[t]*prosumer.c[t]*p for t in prosumer.T)
    prosumer.objective = Objective(rule=MinCost)
    

    return prosumer

