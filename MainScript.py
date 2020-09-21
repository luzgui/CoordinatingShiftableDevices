
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

Ndev=[15]
# Ndev=[15,25,35,45]

for n in Ndev:
    print(n)
    
    # %% generate set of devices
    # Function Appliances() randomly generates a set of pairs (p,d)_n with values in between [1,p_max] [1, d_max]
    p_max=6
    d_max=10
    Devices=Appliances(n, p_max, d_max)
    # print(Devices)
    
    p=Devices[0]
    d=Devices[1]
    
    # D=[p,d]
    # print(pd.DataFrame(D))
    
    #BUG temporary fix
    d[0]=max(d)
    
    ####
    # print(d)
    
    # Plotting a scatter with the distribution of devices
    DevScat(p,d,ResultsFolder,n)
    
    p0=dict(enumerate(p))
    d0=dict(enumerate(d))
    
    nI=len(p)
    
    PD=list((p[i],d[i]) for i in range(len(p)))
    # sorting agents
    
    # Sorting is performed based on consumed power (bigger rated power machines are schedulled first)
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
    
    # PV sizing rule:
    # Due to feasibility reasons we are affecting the total PV installed capacity by a factor of 1.2. This value was determined             
    # experimentally. If f=1, which means a installed capacity producing a quantity of energy equal to the energy consumed by 
    # all the devices together, optimization would run into feasibility issues.
    
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
        
    
    # %%Solver
    opt = SolverFactory('gurobi')
    # opt.options['MIPGap'] = 1e-2
    opt.options['MIPGapAbs'] = 1e-2
    # opt.options['MIPFocus'] = 3
    
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

    # There will be log and solution in YAML, CSV files in Results folder
    SolFile_yaml=os.path.join(CPFolder, 'CP_Sol_Ns_' + str(len(p)) + '.yaml')
    SolFile_csv=os.path.join(CPFolder, 'CP_Sol_Ns_' + str(len(p)) + '.csv')
    
    LogFile=os.path.join(CPFolder, 'CP_Log_Ns_' + str(len(p)) + '.yaml')
    
    Results=opt.solve(prosumer,tee=True, keepfiles=True, logfile=LogFile, solnfile=SolFile_yaml)
    
    #Write results to a .mat file for further processing
    name='CP'
    get_Results_C(prosumer,Results,Ppv,PVcap, n,miu,p,d, ResultsFolder, name)
    #PlotResults
    PlotFunc_Central(prosumer, Ppv, n, ResultsFolder)
    
    # %%
    # 
    ##############################################################################
    ##### MULTI-AGENT #####
    ##############################################################################
    opt.options['MIPGapAbs'] = 1e-10
    R=[];Com=[];X=[];Y=[];P=[]  
    M=[]
    
    # Agents can be queued in different ways (uncomment line):
    # Random queue
    #Iagent=random.sample(range(len(d)),len(d))
    # Sorted by the size of its rated power
    Iagent=range(len(d))
    
    for k in Iagent:
        
        AgentModel=Agent(H,d[k],p[k],c,miu)
        Com.append(AgentModel)
        
        # Write files with individual solution (only needed if a specific agent needs to be investigated)
        SolFile_yaml=os.path.join(DPFolder, 'DP_Sol_Ns_' + str(len(p)) + '_Ag_' + str(k) + '.yaml')
        SolFile_csv=os.path.join(DPFolder, 'DP_Sol_Ns_' + str(len(p)) + '_Ag_' + str(k) + '.csv')
        LogFile=os.path.join(DPFolder, 'DP_Log_Ns_' + str(len(p)) + '_Ag_' + str(k) + '.yaml')
        
        
        Results=opt.solve(AgentModel, tee=True, keepfiles=True, logfile=LogFile, solnfile=SolFile_yaml)

        #Lists containing all Model results
        R.append(Results)
        M.append(AgentModel)
        
        # Calculating some quantities        
        power=[value(AgentModel.x[i])*p[k] for i in AgentModel.T]
        P.append(power)
        Pagg=np.asarray(P)
        Pag=Pagg.sum(axis=0)
        
        #Plotting
        
        if k==len(Iagent)-1: #Plotting only at last iteration to get all agents solution
            
        # Plotting
            fig, ax1 = plt.subplots()
            
            fw=14
            
            T = range(0,H)
            color = 'tab:gray'
            color2='tab:orange'
            ax1.set_xlabel('hour of the day', fontsize=fw)
            ax1.set_ylabel('€/kWh', color=color,fontsize=fw)
            ax1.plot(T,np.asarray(list(c.values())), color=color,linestyle='dashed')
            ax1.tick_params(axis='y',labelsize=fw)
            ax1.set_title('DP N=%i' %n)
            div=12
            L=np.linspace(0,H,div,endpoint=False)
            ax1.set_xticks(L)
            ax1.set_xticklabels(np.linspace(0,24,div,dtype=int,endpoint=False),fontsize=fw)

            
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis'
            ax2.set_ylabel('kW', color=color2)  # we already handled the x-label with ax1
            # ax2.plot(T,power)
            
            for i in Iagent:
                ax2.plot(T,P[i])
                
            ax2.plot(T, Pag, color='black',linewidth=3.0) 
            # ax2.tick_params(axis='y', labelcolor=color)

            # ax2.set_xlabel('hour of the day')
            ax2.set_ylabel('kW', color=color2,fontsize=fw)
            ax2.plot(T,Ppv, color='tab:orange')
            ax2.tick_params(axis='y', labelsize=fw)
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            file=ResultsFolder + '/DP_N_%i' %len(Iagent)
            plt.savefig(file,dpi=300)

        Pag_dict=dict(enumerate(Pag))
        Ppv_dict=dict(enumerate(Ppv))
        
        ## Tariff Update: tariff is incraesed in timeslots in which there is more load
        ## This tarif is sent to the nex agent
        for k in range(H):
            c[k]=c0[k]+0.5*(Pag_dict[k]/PVcap)*TarS
    
    #Write results
    ModelName='DP'
    get_Results_D(M,R, c, Ppv,PVcap, n,miu,p,d, ResultsFolder, ModelName)
    
# Getting a dataframe wit comaprison of all solution .mat files existing in ResultsFolder
df_R=Calc_Tables_mat(ResultsFolder)


# df_R_Server=Calc_Tables_mat('/home/omega/Documents/FCUL/Projects/CoordinatingShiftableDevices/Data/Results_IST/Results')
# PlotCompare(df_R_Server,Ndev,ResultsFolder)
