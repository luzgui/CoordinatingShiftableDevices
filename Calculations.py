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
import os
# from MainScript import n
import scipy.io as sio

cwd = os.getcwd()
DataFolder=cwd + '/Data'
DPFolder=DataFolder + '/DP_Results'
CPFolder=DataFolder + '/CP_Results'
ResultsFolder=DataFolder + '/Results'


# Problem  time data
dt=10 #discretization
H=int((24*60)/dt) # Time horizon
miu=dt/60 #power-energy convertion

def Calcs_Tables(n,DPFolder,CPFolder, ResultsFolder):
    'Makes tables from files outputed from post_process()'

    #Files names import - get al,l the csv files
    Sol_files_DP = [DPFolder + '/' + f for f in listdir(DPFolder) if isfile(join(DPFolder, f)) \
                    and '.csv' in f] 
    Sol_files_DP=[f for f in Sol_files_DP if ('Ns_'+str(n)) in f]    
        
    Sol_files_CP = [CPFolder + '/' + f for f in listdir(CPFolder) if isfile(join(CPFolder, f)) \
                    and '.csv' in f]
    Sol_files_CP=[f for f in Sol_files_CP if ('Ns_'+str(n)) in f]     
    
    #Import CP solutions
    df_csv=pd.read_csv(Sol_files_CP[0], header=None)  
    df=pd.DataFrame([df_csv[1].values],columns=df_csv[0].values)
    
    #Import DP Solutions and construct comparison dataframe
    df_DP=pd.DataFrame(columns=df_csv[0].values)
    for k in Sol_files_DP:
        print(k)
        df_temp=pd.read_csv(k, header=None)
        df_temp=pd.DataFrame([df_temp[1].values],columns=df_temp[0].values)
        # vals=df_temp[1]
        df_DP=pd.concat([df_DP,df_temp],ignore_index=True)
    
    #Convert to floats
    df_DP['Objective']=df_DP['Objective'].astype(float); 
    df_DP['Wall Time']=df_DP['Wall Time'].astype(float); 
    #Compute Total time and Total cost
    df_DP_Total=pd.DataFrame([['DP '+str(n)+' Devices',df_DP.Objective.sum(), df_DP['Wall Time'].sum(), 'optimal']], columns=df_csv[0].values)
    df_DP=pd.concat([df_DP,df_DP_Total],ignore_index=True) 
    #Comparison dataframe
    df=pd.concat([df,df_DP_Total],ignore_index=True) 
    
    
    df.to_csv(ResultsFolder + '/'+'df'+str(n)+'.csv')



def Calc_Tables_mat(ResultsFolder):
     
     # import files
     files=[f for f in listdir(ResultsFolder) if '.mat' in f]
     df_temp=pd.DataFrame()
     df=pd.DataFrame()
     
     for file in files:
         file=ResultsFolder + '/' + file
         Results=sio.loadmat(file)
         df_temp['Model']=Results['Model']
         df_temp['N']=len(Results['P'])
         df_temp['Objective']=Results['Objective']
         df_temp['Objective_T']=Results['Objective_Trans']
         df_temp['Wall_Time']=Results['Wall Time']
         
         df_temp['Tshift']=Results['Tshift']
         df_temp['Txcess']=Results['Txcess']
         df_temp['SSR']=Results['SSR']
         df_temp['SCR']=Results['SCR']
         
         df=df.append(df_temp, ignore_index=True)
     return df
         
         
