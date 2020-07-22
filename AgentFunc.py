from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory


## SINGLE AGENT PROBLEM ##

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


## CENTRALIZED PROBLEM ##

def Agent_C(H,nI,d0,p0,c,miu,Viol,Ppv):  

    prosumer = ConcreteModel()
    # SETS
    prosumer.T = RangeSet(0,H-1)
    
    prosumer.I = RangeSet(0,nI-1)
    
    
    prosumer.c=Param(prosumer.T, initialize=c)
    
    prosumer.p=Param(prosumer.I,initialize=p0)
    
    prosumer.d=Param(prosumer.I,initialize=d0)
    
    # prosumer.miu=Param(prosumer.I, initialize=miu)
    
    def BuildTs(model,nI):
        for i in range(nI):
                return range(model.d[i],H-model.d[i])
    prosumer.Ts = Set(initialize=BuildTs(prosumer,nI))        
    
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
    #    for t in prosumer.Ts:
    #        if t >= prosumer.d[i] or t <= H-prosumer.d[i]:
                return sum(prosumer.x[i,t+k]for k in range(0,prosumer.d[i]))\
            >= prosumer.d[i]*prosumer.y[i,t]
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