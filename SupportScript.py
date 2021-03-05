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


# Getting a dataframe wit comparison of all solution .mat files existing in ResultsFolder
# AppsfilesNames=['AppsList_0','AppsList_1','AppsList_2','AppsList_3','AppsList_4','AppsList_5','AppsList_6','AppsList_7','AppsList_8','AppsList_9','AppsList_10','AppsList_11','AppsList_12','AppsList_13','AppsList_14','AppsList_15','AppsList_16','AppsList_17','AppsList_18','AppsList_19']

# AppsfilesNames=['AppsList_2','AppsList_3',
# 'AppsList_4',
# 'AppsList_8',
# 'AppsList_10',
# 'AppsList_11',
# 'AppsList_12',
# 'AppsList_17',
# 'AppsList_18',
# 'AppsList_19']

# AppsfilesNames=['AppsList_3','AppsList_4','AppsList_8']


#%% PlotCompare
df_R=Calc_Tables_mat(ResultsFolder)

from PlotFunc import *

# full anmes of files Appsfiles=['AppsList_0','AppsList_1','AppsList_2','AppsList_3','AppsList_4','AppsList_5','AppsList_6','AppsList_7','AppsList_8','AppsList_9','AppsList_10','AppsList_11','AppsList_12','AppsList_13','AppsList_14','AppsList_15','AppsList_16''AppsList_17','AppsList_18','AppsList_19']
# Ndev=[25]
# Appsfiles=['AppsList_2.csv']
# Appsfiles=['AppsList_10.csv']



#This is the final apps list
AppsfilesNames=['AppsList_3','AppsList_4','AppsList_8','AppsList_10','AppsList_11','AppsList_12','AppsList_17','AppsList_18']



# # AppsfilesNames=['AppsList_3','AppsList_4','AppsList_8','AppsList_10','AppsList_11','AppsList_12']
# # AppsfilesNames=['AppsList_3','AppsList_8','AppsList_10','AppsList_11','AppsList_12']


#Creates figures 5,8,9 of the manuscript
alpha=0.2
fw=14
[M_CP, M_Rand, M_Sort, Min_CP, Min_Rand, Min_Sorted, Poly]=PlotCompare(df_R,ResultsFolder, AppsfilesNames, AppsFolder + '/DevMean.csv','Mean', alpha, 2.5,fw)

#%% AppsPlott
from PlotFunc import *
color = 'k'
color2='tab:orange'
fw=40
lw=4

#We get the initial tariff that is the same for all, since it only depends on the resource
Tar=sio.loadmat(MatfilesDP[0])
Tar=Tar['Tar']
TarInit=Tar[0,:]



#DP Problem
MatfilesDP=[ResultsFolder + '/DP_Sorted_AppsList_10_25.mat',ResultsFolder +'/DP_Sorted_AppsList_10_125.mat']
# MatfilesDP=[ResultsFolder + '/DP_Random_AppsList_10_25.mat',ResultsFolder +'/DP_Random_AppsList_10_125.mat']
ProfilePlot(MatfilesDP,TarInit, fw,lw, color,color2)

#Centralizied problem
MatfilesCP=[ResultsFolder + '/CP_AppsList_10_25.mat',ResultsFolder +'/CP_AppsList_10_125.mat']
ProfilePlot(MatfilesCP,TarInit, fw,lw, color,color2)


MatfilesNOc=[ResultsFolder + '/Nocord_Random_AppsList_10_25.mat',ResultsFolder +'/DP_Sorted_AppsList_10_125.mat']
# MatfilesDP=[ResultsFolder + '/DP_Random_AppsList_10_25.mat',ResultsFolder +'/DP_Random_AppsList_10_125.mat']
ProfilePlot(MatfilesNOc,TarInit, fw,lw, color,color2)


# df_R=Calc_Tables_mat('/home/omega/Documents/FCUL/Projects/CoordinatingShiftableDevices/ResultsNew')

#To use on local computer after download results
# df_R_Server=Calc_Tables_mat('/home/omega/Documents/FCUL/Projects/CoordinatingShiftableDevices/Data/Results_IST/Results',Appsfiles)
# PlotCompare(df_R_Server,ResultsFolder)

#Comapring 20 appsList
# from PlotFunc import *
# from Calculations import *
# Appsfiles=['AppsList_0','AppsList_1','AppsList_2','AppsList_3','AppsList_4','AppsList_5','AppsList_6','AppsList_7','AppsList_8','AppsList_9','AppsList_10','AppsList_11','AppsList_12','AppsList_13','AppsList_14','AppsList_15','AppsList_16','AppsList_17','AppsList_18','AppsList_19']
# df_125=Calc_Tables_mat('/home/omega/Documents/FCUL/Projects/CoordinatingShiftableDevices/Data/Results_IST/ResultsFullAppsList/AppsList_MAT', Appsfiles)
# df_125['Wall_Time']=df_125['Wall_Time'].astype(float)
# #Check the first 10 with lower walltime
# df_FullList['Wall_Time']=df_FullList['Wall_Time'].astype(float)

# fig = plt.figure()
# plt.plot(df_125['Wall_Time'].values)




# Plotting all
# from PlotFunc import *
# from Calculations import *

# Folder='/home/omega/Documents/FCUL/Projects/CoordinatingShiftableDevices/Data/Results_IST/Results_12Oct'

# Table=Calc_Tables_mat(Folder,Appsfiles)
# PlotCompare(Table,Folder, Appsfiles)

# ResultsFolder='/home/omega/Documents/FCUL/Projects/CoordinatingShiftableDevices/Data/Results_IST/Results-11_10/Results'

#Simple Plots
# from PlotFunc import *
# fw=15
# color='k'
# color2='gold'
# SimplePlot(0,PpvNorm,0,H,fw,color,color2,'Simple')


# x=np.linspace(25,125,11,dtype=int)
# SimplePlot(x,df_DP_Sorted.PVcap,df_DP_Sorted.Tshift,H,fw,color,color2,'TwoAxis')
