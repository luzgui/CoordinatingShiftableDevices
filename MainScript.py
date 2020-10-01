
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

import re 

cwd = os.getcwd()
#general data folder
DataFolder=cwd + '/Data'
#DP results written files
DPFolder=DataFolder + '/DP_Results'
# CP Results written files
CPFolder=DataFolder + '/CP_Results'
# Paper Results
ResultsFolder=DataFolder + '/Results'
#CSV Apps folder
AppsFolder=DataFolder + '/Apps_List'

# Problem  time data
dt=10 #discretization
H=int((24*60)/dt) # Time horizon
miu=dt/60 #power-energy convertion

# %% generate set of devices
# Function Appliances() randomly generates a set of pairs (p,d)_n with values in between [1,p_max] [1, d_max]

n_max=125
p_max=6
d_max=10
N_max=20 # number of files to generate/sets of appliances
## Using several sets of appliances
#Uncoment only when want to generate new devices list
# Appliances(N_max, n_max, p_max, d_max, AppsFolder)

# Generate the sequence of appliances numbers [15,125]
# Ndev=125
Ndev=[14]

# Ndev=np.linspace(15,n_max,12,dtype=int)

#Using just one set of appliances
# FileName='devices_list.csv'
# DevicesFull=pd.read_csv(DataFolder + '/' + FileName)
# DevicesFull = DevicesFull.rename(columns={'Unnamed: 0': 'ind'})

# Import the full set of devices
Appsfiles=[f for f in listdir(AppsFolder)]
#Sort the files
Appsfiles.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

for afiles in Appsfiles:
    print(afiles)
    # Important: run will determine the names of files
    run=re.findall(r"\d+", afiles)
    RunFile='_AppsList_' + run[0]
    print(run[0])
    DevicesFull=pd.read_csv(AppsFolder+ '/'+ afiles)
    DevicesFull = DevicesFull.rename(columns={'Unnamed: 0': 'ind'})
    # Plotting a scatter with the distribution of devices
    # power=DevicesFull['Power'];duration=DevicesFull['Duration']
    # RunFile='run'+ run[0]
    # DevScat(power,duration,ResultsFolder,len(DevicesFull),RunFile)

# %%
    for n in Ndev:
        print(n)
        
        #Slice first n devices
        Devices=DevicesFull.head(n).copy()
        Devices_original=DevicesFull.head(n)
    
        # BUG fix: 
        #identify the index of the first maximum duration device
        idmax=Devices['Duration'].idxmax()
        #swap the row with the biggest duration with the first element of dataframe
        firstmax=Devices.iloc[idmax,:]
        first=Devices.iloc[0,:]
    
        Devices.loc[idmax,['ind','Power','Duration']]=[int(first['ind']),float(first['Power']),float(first['Duration'])]
        Devices.loc[0,['ind','Power','Duration']]=[int(firstmax['ind']),float(firstmax['Power']),float(firstmax['Duration'])]
        power=Devices['Power'].tolist()
        duration=Devices['Duration'].tolist()
        
        p0=dict(enumerate(power))
        dd0=[]
        for k in duration:
            dd0.append(int(k))
        d0=dict(enumerate(dd0))
    
        # %% Sorting devices
        # Sorting is performed based on consumed power (bigger rated power machines are schedulled first)
        Devices_Sort=Devices.sort_values(by='Power',ascending=False)
        Devices_Sort=Devices_Sort.reset_index(drop=True)
        
        p=Devices_Sort['Power'].tolist()
        d=Devices_Sort['Duration'].tolist()
        #convert to integer
        dd=[]
        for k in d:
            dd.append(int(k))
        d=dd
    
        nI=len(p)
        
        
        # %% Calculate energy consumption of all devices
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
        
        # %% Centralized
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
        SolFile_yaml=os.path.join(CPFolder, 'CP_Sol_Ns_' + str(len(p)) + RunFile + '.yaml')
        SolFile_csv=os.path.join(CPFolder, 'CP_Sol_Ns_' + str(len(p)) + RunFile + '.csv')
        
        LogFile=os.path.join(CPFolder, 'CP_Log_Ns_' + str(len(p)) + RunFile + '.yaml')
        
        Results=opt.solve(prosumer,tee=True, keepfiles=True, logfile=LogFile, solnfile=SolFile_yaml)
        
        #Write results to a .mat file for further processing
        name='CP'+ RunFile
        get_Results_C(prosumer,Results,Ppv,PVcap, n,miu,p,d, ResultsFolder, name)
        #PlotResults
        PlotFunc_Central(prosumer, Ppv, n, ResultsFolder,RunFile)
        
        # %% Decentralized
        # 
        ##############################################################################
        ##### MULTI-AGENT #####
        ##############################################################################
        opt.options['MIPGapAbs'] = 1e-10
        R=[];Com=[];X=[];Y=[];P=[]  
        M=[]
        
        # Agents can be queued in different ways (uncomment line):
        # Random queue
        # Iagent=random.sample(range(n),n)
        # ModelSort='_Random'
        
        # Sorted by the size of its rated power
        #This was previsously sorted!!!
        Iagent=range(n)
        ModelSort='_Sorted'
        
        for k in Iagent:
            # print(k)
            
            AgentModel=Agent(H,d[k],p[k],c,miu)
            Com.append(AgentModel)
            
            # Write files with individual solution (only needed if a specific agent needs to be investigated)
            SolFile_yaml=os.path.join(DPFolder, 'DP_Sol_Ns_' + str(len(p)) + '_Ag_' + str(k) + ModelSort + RunFile + '.yaml')
            SolFile_csv=os.path.join(DPFolder, 'DP_Sol_Ns_' + str(len(p)) + '_Ag_' + str(k) + ModelSort + RunFile + '.csv')
            LogFile=os.path.join(DPFolder, 'DP_Log_Ns_' + str(len(p)) + '_Ag_' + str(k) + ModelSort + RunFile + '.yaml')
            
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
                file=ResultsFolder + '/DP_N_%i' %len(Iagent) + ModelSort + RunFile
                plt.savefig(file,dpi=300)
    
            Pag_dict=dict(enumerate(Pag))
            Ppv_dict=dict(enumerate(Ppv))
            
            ## Tariff Update: tariff is increased in timeslots in which there is more load
            ## This tarif is sent to the nex agent
            for k in range(H):
                c[k]=c0[k]+0.5*(Pag_dict[k]/PVcap)*TarS
        
        #Write results
        ModelName='DP'+ ModelSort + RunFile
        get_Results_D(M,R, c, Ppv,PVcap, n,miu,p,d, ResultsFolder, ModelName)
    
# Getting a dataframe wit comaprison of all solution .mat files existing in ResultsFolder
# df_R=Calc_Tables_mat(ResultsFolder)
df_R=Calc_Tables_mat('/home/omega/Documents/FCUL/Projects/CoordinatingShiftableDevices/ResultsNew')
#To use on local computer after download results
# df_R_Server=Calc_Tables_mat('/home/omega/Documents/FCUL/Projects/CoordinatingShiftableDevices/Data/Results_IST/Results')
# PlotCompare(df_R_Server,ResultsFolder)


