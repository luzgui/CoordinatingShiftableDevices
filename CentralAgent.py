from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
from numpy import array
import pandas as pd
from AgentFunc import *
import matplotlib.pyplot as plt
import random
import os
import time

start = time.time()

cwd = os.getcwd()
DataFolder=cwd + '/Data'

n=4

# Time horizon
H=144

#Device characteristics
#p=[3,3,2,1,4,2,2,3,4,2,1,2,3,2,1]
#p=[4,3,2,3,4,2,1,2,4,3,5,6,7,5]
#p=[0.4*k for k in p]; 
#d=[8,4,4,6,5,4,3,4,2,3,2,6,7,4]

p=[4,4,4,3,4,2,1,2,4,3,5,6,7,5]
p=[0.4*k for k in p]; 
p=p*n
d=[12,12,8,6,5,4,3,4,2,3,2,6,7,4]
d=d*n

#p=[4,3,2]
#p=[0.4*k for k in p]; 
#d=[8,4,4]

#d=[4,8,5,2,3,4,7,5,6,2,3,8,7,1,3]

p0=dict(enumerate(p))
d0=dict(enumerate(d))



nI=len(p)

Eshift=sum(p[k]*d[k]*(10/60) for k in range(len(d)));



#Import tarif
#tarFile='/home/omega/Documents/FCUL/PhD/OES/Code/AgentsModel/DataAgents/Tarifa144.csv'
#df = pd.read_csv(tarFile,header=None)
#c=df.to_dict()
#c=c[0]


PVfile=os.path.join(DataFolder,'PV_sim.csv')

dfPV = pd.read_csv(PVfile,header=None)
PpvNorm=dfPV.to_numpy()
# PVcap=3.1
PVcap=3.6*n
Ppv=PVcap*PpvNorm
Epv=sum(Ppv[k]*(10/60) for k in range(H));



#Simple tariff
TarS=0.185;
Tar = np.empty(H)
Tar.fill(TarS)
#Tar.fill(4)

# PV indexed tariff
# for k in range(len(Tar)):
#     Tar[k]=Tar[k]-(TarS*PpvNorm[k]**(1/5))



# Build PV indexed tarif
for k in range(len(Tar)):
#    Tar[k]=Tar[k]-(TarS*PpvNorm[k]**(1/2))
    Tar[k]=Tar[k]-(TarS*PpvNorm[k])
#
c = dict(enumerate(Tar))


## Allowed violation at each timestep
# alpha=0.2
alpha=0
Viol=[alpha*Ppv[k][0] for k in range(len(Ppv))]


prosumer = ConcreteModel()
# SETS
prosumer.T = RangeSet(0,H-1)



prosumer.I = RangeSet(0,nI-1)


prosumer.c=Param(prosumer.T, initialize=c)

prosumer.p=Param(prosumer.I,initialize=p0)

prosumer.d=Param(prosumer.I,initialize=d0)

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
    return sum(sum(prosumer.x[i,t]*prosumer.c[t]*prosumer.p[i]*(10/60) for t in prosumer.T)\
               for i in prosumer.I) 
prosumer.objective = Objective(rule=MinCost)


#    return prosumer

# opt = SolverFactory('cbc')
#opt = SolverFactory('glpk')
opt = SolverFactory('gurobi')

Results=opt.solve(prosumer, tee=True, keepfiles=True)


end = time.time()
print("tempo =",end - start)    

#importing values
P=array([value(prosumer.P[i,t]) for i in prosumer.I for t in prosumer.T])
x=array([value(prosumer.x[i,t]) for i in prosumer.I for t in prosumer.T])

xx=np.reshape(x,(nI,H))
PP=np.reshape(P,(nI,H))

#y=[value(SimpleProsumer.y[i]) for i in SimpleProsumer.T]
P_Raw=np.empty(shape=(nI,H)); P_Raw.fill(0)
for i in prosumer.I:
    P_Raw[i,:]=[value(prosumer.x[i,t])*value(prosumer.p[i]) for t in prosumer.T]

#
#P=np.reshape(P_Raw,(nI,H))
Pag=P_Raw.sum(axis=0)

#Py=[value(SimpleProsumer.y[i])*value(SimpleProsumer.d) for i in SimpleProsumer.T]
#c=[value(SimpleProsumer.c[i]) for i in SimpleProsumer.T]
T=array(list(prosumer.T))



#Td=list(SimpleProsumer.Td)
#Ts=list(SimpleProsumer.Ts)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('hour of the day')
ax1.set_ylabel('euro/kWh', color=color)
ax1.plot(T,np.asarray(list(c.values())), color=color)
ax1.tick_params(axis='y', labelcolor=color)

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

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.savefig('Agent.png',dpi=200)
plt.show()

Etot=sum(Pag[t]*(10/60) for t in range(H));
