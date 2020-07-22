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
from AgentFunc import Agent_C

start = time.time()

cwd = os.getcwd()
DataFolder=cwd + '/Data'


# Problem  time data
dt=10 #discretization
H=int((24*60)/dt) # Time horizon
miu=dt/60 #power-energy convertion


# DEVICES
# [Make function] 

n=2
p=[4,4,4,3,4,2,1,2,4,3,5,6,7,5]
p=[0.4*k for k in p]; 
p=p*n
d=[12,12,8,6,5,4,3,4,2,3,2,6,7,4]
d=d*n

p0=dict(enumerate(p))
d0=dict(enumerate(d))


nI=len(p)


Eshift=sum(p[k]*d[k]*miu for k in range(len(d)));



# PV
PVfile=os.path.join(DataFolder,'PV_sim.csv') #csv for PV
dfPV = pd.read_csv(PVfile,header=None) #create dataframe
PpvNorm=dfPV.to_numpy()
PVcap=3.6*n
Ppv=PVcap*PpvNorm
Epv=sum(Ppv[k]*(miu) for k in range(H));



#Simple tariff
TarS=0.185;
Tar = np.empty(H)
Tar.fill(TarS)
#Tar.fill(4)

# PV indexed tariff
for k in range(len(Tar)):
    Tar[k]=Tar[k]-(TarS*PpvNorm[k]**(1/5))
c = dict(enumerate(Tar))

#Import tarif from file
#tarFile='/home/omega/Documents/FCUL/PhD/OES/Code/AgentsModel/DataAgents/Tarifa144.csv'
#df = pd.read_csv(tarFile,header=None)
#c=df.to_dict()
#c=c[0]

## Allowed violation at each timestep
# alpha=0.2
alpha=0
Viol=[alpha*Ppv[k][0] for k in range(len(Ppv))]



## Problem

prosumer = Agent_C(H,nI,d0,p0,c,miu,Viol,Ppv)
opt = SolverFactory('gurobi')
# opt.options['MIPGap'] = 1e-2
# opt.options['MIPFocus'] = 1
Results=opt.solve(prosumer, tee=True, keepfiles=True)


end = time.time()
print("tempo =",end - start)    


# Results


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
