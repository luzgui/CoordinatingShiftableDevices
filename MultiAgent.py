from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
from AgentFunc import *
import matplotlib.pyplot as plt
import random
from operator import itemgetter
import time
import os

cwd = os.getcwd()
DataFolder=cwd + '/Data'
  
start = time.time()
  
#Import Data Series
#tarFile='/home/omega/Documents/FCUL/PhD/OES/Code/AgentsModel/DataAgents/tarifa.csv'
#tarFileZero='/home/omega/Documents/FCUL/PhD/OES/Code/AgentsModel/DataAgents/tarifazero.csv'


#Import tarif
# tarFile='/home/omega/Documents/FCUL/PhD/OES/Code/AgentsModel/DataAgents/Tarifa144kwh.csv'
# tarFile=os.path.join(DataFolder+'/DataAgents','Tarifa144kwh.csv')
# df = pd.read_csv(tarFile,header=None)
# c=df.to_dict()
# c=c[0]

n=1

#df = pd.read_csv(tarFile,header=None)
#c=df.to_dict()
#
#dfzero=pd.read_csv(tarFileZero,header=None)
#czero=dfzero.to_dict()
 
H=144


PVfile=os.path.join(DataFolder,'PV_sim.csv')
dfPV = pd.read_csv(PVfile,header=None)
PpvNorm=dfPV.to_numpy()

PVcap=3.6*n
Ppv=PVcap*PpvNorm

#Refrence Tarif
TarS=0.185;

Tar = np.empty(H)
Tar.fill(TarS)


#### Build PV indexed tarif
for k in range(len(Tar)):
    Tar[k]=Tar[k]-(TarS*PpvNorm[k]**(1/5))
#    Tar[k]=Tar[k]-(TarS*PpvNorm[k])

Tar0 = Tar
c0= dict(enumerate(Tar0))
c = dict(enumerate(Tar))

#d=[4,8,5,2,3,4,7,4]; 
#p=[6,4,4,6,4,4,6,8];

#p=[3,3,2,1,4,2,2,3,4,2,1,2,3,2,1];
#p=[0.4*k for k in p ]; p=p+p;
#d=[4,8,5,2,3,4,7,5,6,2,3,8,7,1,3];d=d+d; 

p=[4,4,4,3,4,2,1,2,4,3,5,6,7,5]
p=[0.4*k for k in p]; 
p=p*n
d=[12,12,8,6,5,4,3,4,2,3,2,6,7,4]
d=d*n

# p=[3]
# d=[16]


PD=list((p[i],d[i]) for i in range(len(p)))

PDsorted=np.array(sorted(PD,key=itemgetter(0),reverse = True))
p=list(PDsorted[:,0])
d=list(PDsorted[:,1]);d=[int(round(x)) for x in d]


Eshift_i=[p[k]*d[k]*(10/60) for k in range(len(d))];
Eshift=sum(p[k]*d[k]*(10/60) for k in range(len(d)));
Epv=sum(Ppv[k]*(10/60) for k in range(H));



## Algorithm

# opt = SolverFactory('glpk')
opt = SolverFactory('gurobi')
# opt = SolverFactory('cbc')

# opt = SolverFactory('cbc')

R=[];Com=[];X=[];Y=[];P=[]  

#Iagent=random.sample(range(len(d)),len(d))
Iagent=range(len(d))

for k in Iagent:
    
    AgentModel=Agent(H,d[k],p[k],c)
    Com.append(AgentModel)
    Results=opt.solve(AgentModel, tee=True, keepfiles=True)
    R.append(Results)
    # R.append(opt.solve(AgentModel, tee=True, keepfiles=True))
    
#    x=[value(AgentModel.x[i]) for i in AgentModel.T]
#    X.append(x)
    power=[value(AgentModel.x[i])*p[k] for i in AgentModel.T]
    P.append(power)
    Pagg=np.asarray(P)
    Pag=Pagg.sum(axis=0)
    
    fig, ax1 = plt.subplots()
    T = range(0,H)
    color = 'tab:red'
    ax1.set_xlabel('hour of the day')
    ax1.set_ylabel('euro/kWh', color=color)
    ax1.plot(T,np.asarray(list(c.values())), color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    color2= 'tab:green'
    ax2.set_ylabel('kW', color=color)  # we already handled the x-label with ax1
    ax2.plot(T, Pag, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    color = 'tab:red'
    ax2.set_xlabel('hour of the day')
    ax2.set_ylabel('kW', color=color2)
    ax2.plot(T,Ppv, color=color2)
    ax2.tick_params(axis='y', labelcolor=color)


    Pag_dict=dict(enumerate(Pag))
    Ppv_dict=dict(enumerate(Ppv))
    
##    Tariff Update 
    for k in range(H):
        c[k]=c0[k]+0.5*(Pag_dict[k]/PVcap)*TarS
#        c[k]=c0[k]+(Pag_dict[k]/Ppv_dict[k][0])*TarS

end = time.time()

print("tempo =",end - start)    
    
T=np.asarray(list(AgentModel.T))    
#Pagg=np.asarray(P)


#Ploting

# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('hour of the day')
# ax1.set_ylabel('euro/kWh', color=color)
# ax1.plot(T,np.asarray(list(c.values())), color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# color2= 'tab:green'
# ax2.set_ylabel('kW', color=color)  # we already handled the x-label with ax1
# ax2.plot(T, Pag,color='fuchsia')
# ax2.tick_params(axis='y', labelcolor=color)

# #for k in range(len(d)):
# #   ax2.plot(T,Pagg[k,:]) 

# color = 'tab:red'
# ax2.set_xlabel('hour of the day')
# ax2.set_ylabel('kW', color=color2)
# ax2.plot(T,Ppv, color='black')
# ax2.tick_params(axis='y', labelcolor=color)



# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Agent.png',dpi=200)
# plt.show()


Etot=sum(Pag[t]*(10/60) for t in range(H));
