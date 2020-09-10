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
import csv
import pandas as pd
import scipy.io as sio


# def PostProcess(Model, FileNAme):
    
def PostProcess(Model,ModelResults, FileName):    
    'Inputs a model with results loaded and outputs a file with solution in, YMAL or CSV'
    
    SolutionDict={}
    
    SolutionDict['Model']=FileName
    SolutionDict['Objective']=Model.objective()
    SolutionDict['Wall Time']=ModelResults.solver.wall_time
    SolutionDict['Termination Condition']=str(ModelResults.solver.termination_condition)
    
    
    with open(FileName, 'w') as f:
        
        for key in SolutionDict.keys():
            f.write("%s,%s\n"%(key, SolutionDict[key]))

    return SolutionDict


     
def get_Results_C(Model,ModelResults,Ppv,PVcap,n,miu,p,d,ResultsFolder,ModelName):
    'Gets a model from centralized problem and outputs all results to a .mat file'
    
    #importing values
    P=np.array([value(Model.P[i,t]) for i in Model.I for t in Model.T])
    x=np.array([value(Model.x[i,t]) for i in Model.I for t in Model.T])
    c=np.array([value(Model.c[t]) for t in Model.T])
    H=len(c)
    nI=len(Model.I)
    
    xx=np.reshape(x,(nI,H))
    PP=np.reshape(P,(nI,H))
    
    #Solutions for each device
    P_Raw=np.empty(shape=(nI,H)); P_Raw.fill(0)
    for i in Model.I:
        P_Raw[i,:]=[value(Model.x[i,t])*value(Model.p[i]) for t in Model.T]
    
    #Aggregated Load
    Pag=P_Raw.sum(axis=0)
    # Pag=Pag.transpose()
    
    # Transformations
    # xx=pd.DataFrame(xx)
    # xx.index.name='agents'
    
    #Making Table
    # Sol=[]
    
    # P_Raw=pd.DataFrame(P_Raw)
    # P_Raw.index.name='agents'
    
    #Appending values
    # Sol.append(xx)
    # Sol.append(P_Raw)
    # Sol.append(Pag)
    
    #Toal shiftable energy
    P_x=np.zeros(H)
    for k in range(H):
        if Pag[k]>Ppv[k]:
            P_x[k]=Pag[k]-Ppv[k]
        
    #Calculate total excess load
    E_x=sum(P_x[t]*miu for t in range(H));
    
    #Toal shiftable energy
    Eshift=sum(Pag[t]*miu for t in range(H));
    
    #Calculate SSR (gamma% of load is supplied by PV)
    gamma=(Eshift-E_x)/Eshift
    
    
    SolutionDict={}
    
    SolutionDict['Model']=ModelName + '_%i' %n
    SolutionDict['Objective']=Model.objective()
    SolutionDict['Wall Time']=ModelResults.solver.wall_time
    SolutionDict['Termination Condition']=str(ModelResults.solver.termination_condition)
    SolutionDict['P']=P_Raw
    SolutionDict['x']=xx
    SolutionDict['P_ag']=Pag
    SolutionDict['Ppv']=Ppv
    SolutionDict['PVcap']=PVcap
    SolutionDict['Tshift']=Eshift
    SolutionDict['Txcess']=E_x
    SolutionDict['SSR']=gamma
    
    
    FileName = ResultsFolder + '/' + SolutionDict['Model'] + '.mat'
    
    sio.savemat(FileName,SolutionDict)
    
    return SolutionDict




def get_Results_D(ModelArray,ModelResultsArray, Tar, Ppv,PVcap, n,miu,p,d, ResultsFolder, ModelName):
    'Gets an array of Pyomo models (with loaded solution)' 
    'from distributed problem and outputs aggregated results to a .mat file'
    
    P_D=[]
    x_D=[]
    c_D=[]
    SolutionDict={}
    Obj=[]
    Wall_t=[]
    Cond=[]
    
    for m in ModelArray:
        #importing values
        # P=np.array([value(m.P[t]) for t in m.T])
        x=np.array([value(m.x[t]) for t in m.T])
        c=np.array([value(m.c[t]) for t in m.T])
        p=np.array(value(m.p))
        P=[value(m.x[t])*value(m.p) for t in m.T]
        
        obj=m.objective()
        
        H=len(c)
        
        P_D.append(P)
        x_D.append(x)
        c_D.append(c)
        Obj.append(obj)
        
    #Aggregated Load
    Pag=pd.DataFrame(P_D).sum(axis=0)
    Total_Obj=pd.DataFrame(Obj).sum(axis=0)
    
    
    for r in ModelResultsArray:
        wall_time=r.solver.wall_time
        cond=str(r.solver.termination_condition)
        #Wall times
        Wall_t.append(float(wall_time))
        Cond.append(cond)
        
    Total_Wall_t=pd.DataFrame(Wall_t).sum(axis=0)

    

        # Calculating SSR
    # Identify timeslots with violation
    P_x=np.zeros(H)
    for k in range(H):
        if Pag[k]>Ppv[k]:
            P_x[k]=Pag[k]-Ppv[k]
        
    #Calculate total excess load
    E_x=sum(P_x[t]*miu for t in range(H));
    
    #Toal shiftable energy
    Eshift=sum(Pag[t]*miu for t in range(H));
    
    #Calculate SSR (gamma% of load is supplied by PV)
    gamma=(Eshift-E_x)/Eshift

    #Storing values
    SolutionDict['Model']=ModelName + '_%i' %n
    SolutionDict['Objective']=list(Total_Obj)
    SolutionDict['Wall Time']=list(Total_Wall_t)
    SolutionDict['Termination Condition']=Cond
    SolutionDict['P']=P_D
    SolutionDict['x']=x_D
    SolutionDict['P_ag']=list(Pag)
    SolutionDict['Ppv']=Ppv
    SolutionDict['PVcap']=PVcap
    SolutionDict['Tar']=c_D
    SolutionDict['Tshift']=Eshift
    SolutionDict['Dev_p']=p
    SolutionDict['Dev_d']=d
    SolutionDict['Txcess']=E_x
    SolutionDict['SSR']=gamma
    
    FileName = ResultsFolder + '/' + SolutionDict['Model'] + '.mat'
    
    sio.savemat(FileName,SolutionDict)
    
    return SolutionDict




