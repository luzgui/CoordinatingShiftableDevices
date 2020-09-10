
from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import time
from AgentFunc import *
import random
from operator import itemgetter
from PlotFunc import *

import scipy.io as sio
from PostProcess import *
from Calculations import *


cwd = os.getcwd()
#general data folder
DataFolder=cwd + '/Data'
#DP results written files
DPFolder=DataFolder + '/DP_Results'
# CP Results written files
CPFolder=DataFolder + '/CP_Results'
# Paper Results
ResultsFolder=DataFolder + '/Results'


# Problem  time data
dt=10 #discretization
H=int((24*60)/dt) # Time horizon
miu=dt/60 #power-energy convertion


# n=3
# p=[4,4,4,3,4,2,1,2,4,3,5,6,7,5]
# p=[0.4*k for k in p]; 
# p=p*n
# d=[12,12,8,6,5,4,3,4,2,3,2,6,7,4]
# d=d*n

# p=n*[1.6,1.6,1.6,1.2,1.6,0.8,0.4,0.8,1.6,1.2,2.0,2.4,2.8,2.0]
    
#     # p=p*n
# d=n*[12,12,8,6,5,4,3,4,2,3,2,6,7,4]


# DEVICES 
# Number of devices
# Ndev=[14,15,16,17,18,19,20]
Ndev=[14,17,20,23]

# Ndev=[14]
# Ndev=[10,11,12,13,14]

for n in Ndev:
    print(n)
    
    # %% generate set of devices
    Devices=Appliances(n)
    print(Devices)
    p=Devices[0]
    d=Devices[1]
    #BUG temporary fix
    d[0]=max(d)
    ####
    print(d)
    
    p0=dict(enumerate(p))
    d0=dict(enumerate(d))
    
    
    nI=len(p)
    
    PD=list((p[i],d[i]) for i in range(len(p)))
    # sorting agents
    PDsorted=np.array(sorted(PD,key=itemgetter(0),reverse = True))
    
    p=list(PDsorted[:,0])
    d=list(PDsorted[:,1]);d=[int(round(x)) for x in d]
    
    #Calculate energy consumption of all devices
    Eshift_i=[p[k]*d[k]*miu for k in range(len(d))];
    Eshift=sum(p[k]*d[k]*miu for k in range(len(d)));
    
    
    
    # %% PV definition
    PVfile=os.path.join(DataFolder,'PV_sim.csv') #csv for PV
    dfPV = pd.read_csv(PVfile,header=None) #create dataframe
    PpvNorm=dfPV.to_numpy()
    # PVcap=3.6*n
    # PVcap=10
    
    # PV sizing rule: 
    f=1.2 # PV multiplying factor
    PVcap=f*Eshift/sum(PpvNorm[k]*(miu) for k in range(H))
    PVcap=PVcap[0]
    Ppv=PVcap*PpvNorm
    
    
    Epv=sum(Ppv[k]*(miu) for k in range(H));
    
    #PV LCOE
    c_PV=0.04
    
    # %% Tarifs
    #Simple tariff
    TarS=0.185;
    Tar = np.empty(H)
    Tar.fill(TarS)
    #Tar.fill(4)
    
    # PV indexed tariff
    #we subtract the LCOE from the tariff
    for k in range(len(Tar)):
        Tar[k]=Tar[k]-((TarS-c_PV)*PpvNorm[k]**(1/5))
        
    Tar0 = Tar
    c0= dict(enumerate(Tar0))    
    c = dict(enumerate(Tar))
    
    #Import tarif
    # tarFile='/home/omega/Documents/FCUL/PhD/OES/Code/AgentsModel/DataAgents/Tarifa144kwh.csv'
    # tarFile=os.path.join(DataFolder+'/DataAgents','Tarifa144kwh.csv')
    # df = pd.read_csv(tarFile,header=None)
    # c=df.to_dict()
    # c=c[0]
    
    
    
    # %%Solver
    opt = SolverFactory('gurobi')
    
    # %%
    
    ##############################################################################
    ##### CENTRALIZED #####
    ##############################################################################
    
    
    # Allowed violation at each timestep
    alpha=0.2
    alpha=0
    Viol=[alpha*Ppv[k][0] for k in range(len(Ppv))]
    
    # Solving problem
    prosumer = Agent_C(H,nI,d0,p0,c,miu,Viol,Ppv)
    # opt.options['MIPGap'] = 1e-2
    # opt.options['MIPFocus'] = 1
    
    # Results=opt.solve(prosumer, tee=True, keepfiles=True, logfile='log.log')
    
    
    # FileName='CP_Sol_Ns_'+str(len(p))+'.yaml'
    
    SolFile_yaml=os.path.join(CPFolder, 'CP_Sol_Ns_' + str(len(p)) + '.yaml')
    SolFile_csv=os.path.join(CPFolder, 'CP_Sol_Ns_' + str(len(p)) + '.csv')
    
    LogFile=os.path.join(CPFolder, 'CP_Log_Ns_' + str(len(p)) + '.yaml')
    
    Results=opt.solve(prosumer,tee=True, keepfiles=True, logfile=LogFile, solnfile=SolFile_yaml)
    
    #Write results
    # post.PostProcess(prosumer,Results,SolFile_csv)
    name='CP'
    
    get_Results_C(prosumer,Results,Ppv,PVcap, n,miu,p,d, ResultsFolder, name)
    #PlotResults
    PlotFunc_Central(prosumer, Ppv, n, ResultsFolder)
    
    
    
    # %%
    # 
    ##############################################################################
    ##### MULTI-AGENT #####
    ##############################################################################
    
#     R=[];Com=[];X=[];Y=[];P=[]  
#     M=[]
    
#     #Iagent=random.sample(range(len(d)),len(d))
#     Iagent=range(len(d))
    
#     for k in Iagent:
        
#         AgentModel=Agent(H,d[k],p[k],c,miu)
#         Com.append(AgentModel)
    
#         SolFile_yaml=os.path.join(DPFolder, 'DP_Sol_Ns_' + str(len(p)) + '_Ag_' + str(k) + '.yaml')
#         SolFile_csv=os.path.join(DPFolder, 'DP_Sol_Ns_' + str(len(p)) + '_Ag_' + str(k) + '.csv')
#         LogFile=os.path.join(DPFolder, 'DP_Log_Ns_' + str(len(p)) + '_Ag_' + str(k) + '.yaml')
        
#         Results=opt.solve(AgentModel, tee=True, keepfiles=True, logfile=LogFile, solnfile=SolFile_yaml)
#         # post.PostProcess(AgentModel,Results,SolFile_csv)
        
#         #Lists containing all Model results
#         R.append(Results)
#         M.append(AgentModel)
#         # R.append(opt.solve(AgentModel, tee=True, keepfiles=True))
        
#         # x=[value(AgentModel.x[i]) for i in AgentModel.T]
#         # X.append(x)
        
#         power=[value(AgentModel.x[i])*p[k] for i in AgentModel.T]
#         P.append(power)
#         Pagg=np.asarray(P)
#         Pag=Pagg.sum(axis=0)
        
#         #Plotting
        
#         if k==len(Iagent)-1:
            
#         # Plotting
#             fig, ax1 = plt.subplots()
            
#             T = range(0,H)
#             color = 'tab:red'
#             ax1.set_xlabel('hour of the day')
#             ax1.set_ylabel('euro/kWh', color=color)
#             ax1.plot(T,np.asarray(list(c.values())), color=color)
#             ax1.tick_params(axis='y', labelcolor=color)
#             ax1.set_title('DP Nagents: %i' %n)
#             ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
#             color = 'tab:blue'
#             color2= 'tab:green'
#             ax2.set_ylabel('kW', color=color)  # we already handled the x-label with ax1
#             # ax2.plot(T,power)
            
#             for i in Iagent:
#                 ax2.plot(T,P[i])
                
#             ax2.plot(T, Pag, color='black',linewidth=3.0) 
#             ax2.tick_params(axis='y', labelcolor=color)
            
#             color = 'tab:red'
#             ax2.set_xlabel('hour of the day')
#             ax2.set_ylabel('kW', color=color2)
#             ax2.plot(T,Ppv, color=color2)
#             ax2.tick_params(axis='y', labelcolor=color)
#             fig.tight_layout()  # otherwise the right y-label is slightly clipped
#             file=ResultsFolder + '/DP_N_%i' %len(Iagent)
#             plt.savefig(file,dpi=200)

#         Pag_dict=dict(enumerate(Pag))
#         Ppv_dict=dict(enumerate(Ppv))
        
#         ## Tariff Update 
#         for k in range(H):
#             c[k]=c0[k]+0.5*(Pag_dict[k]/PVcap)*TarS
#         #   c[k]=c0[k]+(Pag_dict[k]/Ppv_dict[k][0])*TarS
    
#     #Write results
#     ModelName='DP'
#     get_Results_D(M,R, c, Ppv,PVcap, n,miu,p,d, ResultsFolder, ModelName)
    
#     T=np.asarray(list(AgentModel.T))    
#     Etot=sum(Pag[t]*miu for t in range(H));

df_R=Calc_Tables_mat(ResultsFolder)



