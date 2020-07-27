from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd
from AgentFunc import *
import matplotlib.pyplot as plt
import random
from operator import itemgetter
from os import listdir
from os.path import isfile, join
import yaml
import calliope as cal
import pandas as pd


cwd = os.getcwd()
DataFolder=cwd + '/Data'
DPFolder=DataFolder + '/DP_Results'
CPFolder=DataFolder + '/CP_Results'


# Problem  time data
dt=10 #discretization
H=int((24*60)/dt) # Time horizon
miu=dt/60 #power-energy convertion



#Files names import - get al,l the csv files
Sol_files_DP = [DPFolder + '/' + f for f in listdir(DPFolder) if isfile(join(DPFolder, f)) and '.csv' in f] 
Sol_files_CP = [CPFolder + '/' + f for f in listdir(CPFolder) if isfile(join(CPFolder, f)) and '.csv' in f] 


#Import CP solutions
df_csv=pd.read_csv(Sol_files_CP[0], header=None)  
df=pd.DataFrame([df_csv[1].values],columns=df_csv[0].values)

#Import DP Solutions and construct comparison dataframe
df_DP=pd.DataFrame(columns=df_csv[0].values)
for k in Sol_files_DP:
    df_temp=pd.read_csv(k, header=None)
    df_temp=pd.DataFrame([df_temp[1].values],columns=df_temp[0].values)
    # vals=df_temp[1]
    df_DP=pd.concat([df_DP,df_temp],ignore_index=True)

#Convert to floats
df_DP['Objective']=df_DP['Objective'].astype(float); 
df_DP['Wall Time']=df_DP['Wall Time'].astype(float); 
#Compute Total time and Total cost
df_DP_Total=pd.DataFrame([['DP 14 Devices',df_DP.Objective.sum(), df_DP['Wall Time'].sum(), 'optimal']], columns=df_csv[0].values)
df_DP=pd.concat([df_DP,df_DP_Total],ignore_index=True) 
#Comparison dataframe
df=pd.concat([df,df_DP_Total],ignore_index=True) 




#Identify timeslots with violation
P_PV_ex= np.zeros(H)
for k in range(H):
    if Pag[k]>Ppv[k]:
        P_PV_ex[k]=Pag[k]-Ppv[k]
    

#Calculate all that energy
E_T_PV=sum(P_PV_ex[t]*(10/60) for t in range(H));

#Toal shiftable energy
Etot=sum(Pag[t]*(10/60) for t in range(H));

E_F=(Etot-E_T_PV)/Etot

print(Etot)
print(E_T_PV)
print(E_F)

