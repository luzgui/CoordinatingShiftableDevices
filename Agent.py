from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory

prosumer = AbstractModel()
#SETS

prosumer.H=Param()
#timeslot set
prosumer.T = RangeSet(0,prosumer.H-1)



#PARAMETERS

# Device duration
prosumer.d = Param()

prosumer.Td = RangeSet(0, prosumer.d-1)
prosumer.Ts = RangeSet(0,prosumer.H-prosumer.d-1)

# Power consumed set
prosumer.p=Param()

# dynamic tariff (Must be imported)
prosumer.c=Param(prosumer.T)
# VARIABLES

# Starting variable
prosumer.y = Var(prosumer.T,domain=Binary)
# Activity variable
prosumer.x = Var(prosumer.T,domain=Binary)
# Power cosumption



#
## CONSTRAINTS
def Consty(prosumer,t):
    return sum(prosumer.y[t] for t in prosumer.T) == 1
prosumer.y_constraint = Constraint(prosumer.T,rule=Consty)

#def Constxy(prosumer,t):
#    i=1
#    for i in range(0,value(prosumer.d)):
#              return sum(prosumer.x[t+i] for t in prosumer.Ts) \
#                             <= prosumer.d*(1-prosumer.y[t])
#    print(i)
#    i+=1
#prosumer.xy_constraint = Constraint(prosumer.Ts,rule=Constxy) 

def Constxy(prosumer,t):
    
    return sum(prosumer.x[t+i]for i in range(0,d))>= prosumer.d*prosumer.y[t]
    prosumer.xy_constraint = Constraint(prosumer.Ts,rule=Constxy)

def Constx(prosumer,t):
    return sum(prosumer.x[t] for t in prosumer.T) == prosumer.d
prosumer.x_constraint = Constraint(prosumer.T,rule=Constx) 

#OBJECTIVE
def MinCost(prosumer):
    return sum(prosumer.x[t]*prosumer.c[t]*prosumer.p for t in prosumer.T)
prosumer.objective = Objective(rule=MinCost)
#Solving the model


##########




SimpleProsumer = prosumer.create_instance("prosumerData2.dat")    

opt = SolverFactory('glpk')
opt.solve(SimpleProsumer, tee=True, keepfiles=True)



#importing values
x=[value(SimpleProsumer.x[i]) for i in SimpleProsumer.T]
y=[value(SimpleProsumer.y[i]) for i in SimpleProsumer.T]
P=[value(SimpleProsumer.x[i])*value(SimpleProsumer.d) for i in SimpleProsumer.T]
Py=[value(SimpleProsumer.y[i])*value(SimpleProsumer.d) for i in SimpleProsumer.T]
c=[value(SimpleProsumer.c[i]) for i in SimpleProsumer.T]
T=list(SimpleProsumer.T)
Td=list(SimpleProsumer.Td)
Ts=list(SimpleProsumer.Ts)



#Ploting
import matplotlib.pyplot as plt
import numpy as np

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('hour of the day')
ax1.set_ylabel('euro/kWh', color=color)
ax1.plot(T,c, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
color2= 'tab:green'
ax2.set_ylabel('kW', color=color)  # we already handled the x-label with ax1
ax2.plot(T, P, color=color)
ax2.plot(T, Py, color=color2)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('SimpleProsumer.png',dpi=200)
plt.show()