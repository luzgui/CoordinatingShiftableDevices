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
from PlotFunc import *

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
#Figures from paper
PicsFolder=DataFolder + '/Pics'

# Getting a dataframe wit comparison of all solution .mat files existing in ResultsFolder
df_R=Calc_Tables_mat(ResultsFolder)

# Full name of files Appsfiles=['AppsList_0','AppsList_1','AppsList_2','AppsList_3','AppsList_4','AppsList_5','AppsList_6','AppsList_7','AppsList_8','AppsList_9','AppsList_10','AppsList_11','AppsList_12','AppsList_13','AppsList_14','AppsList_15','AppsList_16''AppsList_17','AppsList_18','AppsList_19']

#This is the final apps list used in the paper
AppsfilesNames=[
'AppsList_3',
'AppsList_4',
'AppsList_8',
'AppsList_10',
'AppsList_11',
'AppsList_12',
'AppsList_17',
'AppsList_18']

#Creates figures 5,8,9 of the manuscript
alpha=0.2
fw=20
[M_CP, M_Rand, M_Sort, Min_CP, Min_Rand, Min_Sorted, Poly]=PlotCompare(df_R,ResultsFolder, AppsfilesNames, AppsFolder + '/DevMean.csv','Mean', alpha, 2.5,fw)

#%% AppsPlott: creates figures 4, 6,7
from PlotFunc import *
color = 'k'
color2='tab:orange'
fw=40 #font weight
lw=4 #line weight


#We get the initial tariff that is the same for all, since it only depends on the resource
Tar=sio.loadmat(MatfilesDP[0])
Tar=Tar['Tar']
TarInit=Tar[0,:]

#Output the plots
#DP Problem - Sorted
MatfilesDP=[ResultsFolder + '/DP_Sorted_AppsList_10_25.mat',ResultsFolder +'/DP_Sorted_AppsList_10_125.mat']
ProfilePlot(MatfilesDP,TarInit, fw,lw, color,color2,'DP(Sorted)',PicsFolder + 'apps_DPSort.png')

#DP Problem - Random
MatfilesDP=[ResultsFolder + '/DP_Random_AppsList_10_25.mat',ResultsFolder +'/DP_Random_AppsList_10_125.mat']
ProfilePlot(MatfilesDP,TarInit, fw,lw, color,color2,'DP(Random)',PicsFolder + 'apps_DPRand.png')

#Centralized problem
MatfilesCP=[ResultsFolder + '/CP_AppsList_10_25.mat',ResultsFolder +'/CP_AppsList_10_125.mat']
ProfilePlot(MatfilesCP,TarInit, fw,lw, color,color2,'CP',PicsFolder + 'apps_CP2_xxx.png')


MatfilesNOc=[ResultsFolder + '/Nocord_Random_AppsList_10_25.mat',ResultsFolder +'/DP_Sorted_AppsList_10_125.mat']
# MatfilesDP=[ResultsFolder + '/DP_Random_AppsList_10_25.mat',ResultsFolder +'/DP_Random_AppsList_10_125.mat']
ProfilePlot(MatfilesNOc,TarInit, fw,lw, color,color2, 'No Coordination',PicsFolder + 'NoCord_output.png')

