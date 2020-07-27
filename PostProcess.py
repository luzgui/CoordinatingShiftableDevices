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
     