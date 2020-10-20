from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import os



def PlotFunc_Central(Model,Ppv, n, ResultsFolder,RunFile):
    "Model argument is a Pyomo model with results, n: number of agents, Ppv: PV capacity,"
    "Resultsfolder: destination of plots"

    #importing values
    P=array([value(Model.P[i,t]) for i in Model.I for t in Model.T])
    x=array([value(Model.x[i,t]) for i in Model.I for t in Model.T])
    c=array([value(Model.c[t]) for t in Model.T])
    H=len(c)
    nI=len(Model.I)
    
    xx=np.reshape(x,(nI,H))
    PP=np.reshape(P,(nI,H))
    
    P_Raw=np.empty(shape=(nI,H)); P_Raw.fill(0)
    for i in Model.I:
        P_Raw[i,:]=[value(Model.x[i,t])*value(Model.p[i]) for t in Model.T]
    
    Pag=P_Raw.sum(axis=0)
    T=array(list(Model.T))
    
    #make figure
    fig, ax1 = plt.subplots()
    fw=14
    
    color = 'tab:gray'
    color2='tab:orange'
    ax1.set_xlabel('hour of the day', fontsize=fw)
    ax1.set_ylabel('€/kWh', color=color,fontsize=fw)
    ax1.plot(T,np.asarray(list(c)), color=color,linestyle='dashed')
    ax1.tick_params(axis='y',labelsize=fw)
    ax1.set_title('CP N=%i' %n)
    div=12
    L=np.linspace(0,H,div,endpoint=False)
    ax1.set_xticks(L)
    ax1.set_xticklabels(np.linspace(0,24,div,dtype=int,endpoint=False),fontsize=fw)

    
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis'
    ax2.set_ylabel('kW', color=color2)  # we already handled the x-label with ax1
    # ax2.plot(T,power)
    
    for k in range(len(P_Raw)):
       ax2.plot(T,P_Raw[k,:]) 
        
    ax2.plot(T, Pag, color='black',linewidth=3.0) 
    # ax2.tick_params(axis='y', labelcolor=color)

    # ax2.set_xlabel('hour of the day')
    ax2.set_ylabel('kW', color=color2,fontsize=fw)
    ax2.plot(T,Ppv, color='tab:orange')
    ax2.tick_params(axis='y', labelsize=fw)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    file=ResultsFolder + '/CP_N_%i' %n + RunFile
    plt.savefig(file,dpi=300)
    plt.show()

    
    
    
    
    
    
    
    # color = 'tab:red'
    # ax1.set_xlabel('hour of the day',fontsize=fw)
    # ax1.set_ylabel('euro/kWh', color=color,fontsize=fw)
    # ax1.plot(T,np.asarray(list(c)), color=color)
    # ax1.tick_params(axis='y', labelcolor=color,labelsize=fw)
    
    
    # ax1.set_title('CP N=%i' %n)
    
    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    # color = 'tab:blue'
    # color2= 'tab:green'
    # ax2.set_ylabel('kW', color=color)  # we already handled the x-label with ax1
    
    # for k in range(len(P_Raw)):
    #    ax2.plot(T,P_Raw[k,:]) 
    
    # ax2.plot(T,Pag , color='black',linewidth=3.0)
    # ax2.tick_params(axis='y', labelcolor=color)
    
    # color = 'tab:red'
    # ax2.set_xlabel('hour of the day')
    # ax2.set_ylabel('kW', color=color2)
    # ax2.plot(T,Ppv, color=color2)
    # ax2.tick_params(axis='y', labelcolor=color)
    
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # file=ResultsFolder + '/CP_N_%i' %n
    # plt.savefig(file,dpi=300)
    
    # plt.show()


def DevScat(p,d,ResultsFolder,n,RunFile):
    'Gerate a scatter for the generated devices with points (p,d)=(power, duration)'
    fw=14
    fig = plt.figure()
    #Generate a list of unique points
    points=list(set(zip(p,d))) 
    #Generate a list of point counts
    count=[len([x for x,y in zip(p,d) if x==pp[0] and y==pp[1]]) for pp in points]
    #Now for the plotting:
    plot_x=[i[0] for i in points]
    plot_y=[i[1] for i in points]
    count=np.array(count)
    plt.scatter(plot_x,plot_y,c=count,s=100*count**0.5,cmap='Spectral_r')
    plt.xlabel("power demand (kW)", fontsize=fw)
    plt.ylabel("Duration (nº timeslots)", fontsize=fw)
    plt.colorbar()
    plt.grid(True)
    plt.title('Shiftable devices distribution N=%i' %n, fontsize=fw)
    file=ResultsFolder + '/Scatter_N_%i' %n + '_' + RunFile
    plt.savefig(file,dpi=300)
    plt.show()

#%% Plot Compare

def PlotCompare(df,ResultsFolder, Appsfiles,DevMeanFile,stat,a,linebig):
    'df: data frame with the results'
    'Resultsfolder: images destination'
    'Appsfiles: list of Appsfiles names'
    'DevMeanFile: a path to a CSV with the DevicesList_Mean dataframe (check mainscript) '
    
    DevicesList_Mean=pd.read_csv(DevMeanFile)
    
    df['Wall_Time']=df['Wall_Time'].astype(float)
    
    df_mean_CP = pd.DataFrame(columns=df.columns)
    df_mean_DP = pd.DataFrame(columns=df.columns)
    df_mean_DP_Sorted = pd.DataFrame(columns=df.columns)
    df_mean_DP_Random = pd.DataFrame(columns=df.columns)
    
    # Calculate the means for each N
    
    df=df.loc[df['AppsList'].isin(Appsfiles)] # its important if there are other mat files in folder
    df=df[df.N !=15]
    df_CP=df[df.Model.str.contains('CP', case=True)]
    df_DP=df[df.Model.str.contains('DP', case=True)]
    
    df_DP_Sorted=df[df.Model.str.contains('DP_Sorted', case=True)]
    df_DP_Random=df[df.Model.str.contains('DP_Random', case=True)]
    
    for i in df.N.unique():

            
        df_temp_CP=df_CP.loc[df_CP.N==i]
        df_temp_DP=df_DP.loc[df_DP.N==i]
        df_temp_DP_Sorted=df_DP_Sorted.loc[df_DP_Sorted.N==i]
        df_temp_DP_Random=df_DP_Random.loc[df_DP_Random.N==i]
        # print(df_temp)
        #Incredible sequance of 3 functions applied
        if stat=='Mean':
            df_mean_CP=pd.concat([df_mean_CP,df_temp_CP.mean().to_frame().transpose()])
            df_mean_DP=pd.concat([df_mean_DP,df_temp_DP.mean().to_frame().transpose()])
            df_mean_DP_Sorted=pd.concat([df_mean_DP_Sorted,df_temp_DP_Sorted.mean().to_frame().transpose()])
            df_mean_DP_Random=pd.concat([df_mean_DP_Random,df_temp_DP_Random.mean().to_frame().transpose()])
            MeanType='Mean'
        #median
        elif stat=='Median':
            df_mean_CP=pd.concat([df_mean_CP,df_temp_CP.median().to_frame().transpose()])
            df_mean_DP=pd.concat([df_mean_DP,df_temp_DP.median().to_frame().transpose()])
            df_mean_DP_Sorted=pd.concat([df_mean_DP_Sorted,df_temp_DP_Sorted.median().to_frame().transpose()])
            df_mean_DP_Random=pd.concat([df_mean_DP_Random,df_temp_DP_Random.median().to_frame().transpose()])
            
            MeanType='Median'
        
        
    
    df_mean_CP=df_mean_CP.sort_values(by='N')    
    df_mean_DP=df_mean_DP.sort_values(by='N')   
    df_mean_DP_Sorted=df_mean_DP_Sorted.sort_values(by='N')   
    df_mean_DP_Random=df_mean_DP_Random.sort_values(by='N')   
    
    colorRuns = plt.get_cmap('Pastel2')
    colorMean = plt.get_cmap('tab20b')
    
    import seaborn as sns
    fig, axs = plt.subplots(3, 2)
    # fig, axs = plt.subplots(1, 1)
    #Plot the mean
    
    label=[]
    for k in Appsfiles:
        # print(k)
        # if k in str(df['AppsList']):
        # print('XX '+ k)
        label.append(k)
        
        df_temp=df.loc[df['AppsList']==k]
        # DevMean_temp=DevicesList_Mean.loc[DevicesList_Mean['AppsList']==k]
        # print(df_temp)
        
        df_CP=pd.DataFrame
        df_CP=df_temp[df_temp.Model.str.contains('CP', case=True)]
        df_CP=df_CP.sort_values(by='N')
        df_CP=df_CP.reset_index(drop=True)
        
        df_DP=pd.DataFrame
        df_DP=df_temp[df_temp.Model.str.contains('DP', case=True)]
        df_DP=df_DP.sort_values(by='N')
        df_DP=df_DP.reset_index(drop=True)
        
        df_DP_Sorted=pd.DataFrame
        df_DP_Sorted=df_temp[df_temp.Model.str.contains('DP_Sorted', case=True)]
        df_DP_Sorted=df_DP_Sorted.sort_values(by='N')
        df_DP_Sorted=df_DP_Sorted.reset_index(drop=True)
        
        df_DP_Random=pd.DataFrame
        df_DP_Random=df_temp[df_temp.Model.str.contains('DP_Random', case=True)]
        df_DP_Random=df_DP_Random.sort_values(by='N')
        df_DP_Random=df_DP_Random.reset_index(drop=True)
        
        #%% Sorted + Random plotting
        axs[0,0].plot(df_DP_Sorted['N'],df_DP_Sorted['Wall_Time']/60, color=colorRuns(1))
        axs[0,0].plot(df_DP_Random['N'],df_DP_Random['Wall_Time']/60, color=colorRuns(2))

        axs[0,0].axes.set_xticks(df_DP['N'])
        # axs[0,0].legend(['Sorted','Random'])
        # axs[0,0].grid()
        axs[0,0].axes.set_xlabel('Number of Agents')
        axs[0,0].axes.set_ylabel('Time(min)')
        axs[0,0].set_title('Wall Time (DP)')
    
        axs[1,0].plot(df_CP['N'],df_CP['Wall_Time']/60)
        axs[1,0].axes.set_xticks(df_CP['N'])
        # axs[1,0].legend(label)
        # axs[1,0].grid()
        axs[1,0].axes.set_xlabel('Number of Agents')
        axs[1,0].axes.set_ylabel('Time(min)')
        axs[1,0].set_title('Wall Time (CP)')
        
        Norm_temp=((df_DP['Objective_T']-df_CP['Objective'])/df_CP['Objective'])*100
        Norm_temp_Sorted=((df_DP_Sorted['Objective_T']-df_CP['Objective'])/df_CP['Objective'])*100
        Norm_temp_Random=((df_DP_Random['Objective_T']-df_CP['Objective'])/df_CP['Objective'])*100
        print(Norm_temp_Sorted)
        # Norm2=((df_mean_DP['Objective']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
        axs[0,1].plot(df_DP_Sorted['N'],Norm_temp_Sorted,color='blue')
        axs[0,1].plot(df_DP_Random['N'],Norm_temp_Random,color='red')
        axs[0,1].legend(['Sorted','Random'])
        # axs[0,1].plot(df_CP['N'],df_CP['Objective'])
        axs[0,1].axes.set_xticks(df_DP['N'])
        # axs[0,1].grid()
        axs[0,1].axes.set_xlabel('Number of Agents')
        axs[0,1].axes.set_ylabel('%')
        axs[0,1].set_title('DP objective (trans) relative to CP optimal')
        
        # axs[1,1].plot(N,df_DP['SSR'].astype(float),color='red')
        axs[1,1].plot(df_CP['N'],df_CP['SSR']*100,color='k' )
        axs[1,1].plot(df_DP_Sorted['N'],df_DP_Sorted['SSR']*100, color='blue')
        axs[1,1].plot(df_DP_Random['N'],df_DP_Random['SSR']*100, color='red')
        axs[1,1].legend(['CP-Optimal','Sorted','Random'])
        axs[1,1].axes.set_xticks(df_CP['N'])
        # axs[1,1].grid()
        # axs[1,1].axes.set_xlabel('Number of Agents')
        axs[1,1].axes.set_ylabel('%')
        axs[1,1].set_title('Self-Suficiency Ratio')
        
        # axs[2,0].plot(DevMean_temp['N'],DevMean_temp['m_p'])
        # axs[2,0].plot(DevMean_temp['N'],DevMean_temp['m_d'])
        # axs[2,0].axes.set_xticks(df_CP['N'])
        # # axs[2,0].legend(label)
        # axs[2,0].grid()
        # axs[2,0].axes.set_xlabel('Number of Agents')
        # axs[2,0].axes.set_ylabel('kW')
        
        axs[2,1].plot(df_CP['N'],df_CP['SCR']*100,color='k')
        # axs[2,1].plot(df_DP['N'],df_DP['SCR'])
        axs[2,1].plot(df_DP_Sorted['N'],df_DP_Sorted['SCR']*100, color='blue')
        axs[2,1].plot(df_DP_Random['N'],df_DP_Random['SCR']*100, color='red')
        axs[2,1].legend(['CP-Optimal','Sorted','Random'])
        axs[2,1].axes.set_xticks(df_CP['N'])
        axs[2,1].axes.set_xlabel('Number of Agents')
        axs[2,1].axes.set_ylabel('%')
        axs[2,1].set_title('Self-Consumption Rate')
        axs[2,1].grid()
    
        

    
    axs[0,0].axes.set_xticks(df_mean_CP['N'])
    axs[0,1].axes.set_xticks(df_mean_CP['N'])
    axs[1,0].axes.set_xticks(df_mean_CP['N'])
    axs[1,1].axes.set_xticks(df_mean_CP['N'])
    axs[2,1].axes.set_xticks(df_mean_CP['N'])
    
    axs[0,0].grid()
    axs[0,1].grid()
    axs[1,0].grid()
    axs[1,1].grid()
    axs[2,1].grid()
    plt.tight_layout()        
    plt.show()
            
    # plt.style.use('seaborn-darkgrid')
    fig, axs = plt.subplots(3, 2)
        
    # axs[0,0].plot(df_mean_DP_Sorted['N'],df_mean_DP_Sorted['Wall_Time']/60,color=colorMean(1),linewidth=linebig)
    # axs[0,0].plot(df_mean_DP_Random['N'],df_mean_DP_Random['Wall_Time']/60,color=colorMean(14),linewidth=linebig)
    axs[0,0].plot(df_mean_DP_Sorted['N'],df_mean_DP_Sorted['Wall_Time']/60,color=colorRuns(1),linewidth=linebig)
    axs[0,0].plot(df_mean_DP_Random['N'],df_mean_DP_Random['Wall_Time']/60,color=colorRuns(2),linewidth=linebig)
    axs[0,0].legend(['Sorted-' + MeanType,'Random-'+ MeanType])
    axs[0,0].axes.set_xticks(df_mean_DP['N'])
    axs[0,0].grid()
    # axs[0,0].axes.set_xlabel('Number of Agents')
    # axs[0,0].axes.set_ylabel('Time(min)')
    # axs[0,0].set_title('Wall Time (DP)')
    
    axs[1,0].plot(df_mean_CP['N'],df_mean_CP['Wall_Time']/60,color='k')
    axs[1,0].legend([MeanType],loc='upper left')
    axs[1,0].axes.set_xticks(df_mean_CP['N'])
    axs[1,0].grid()
    axs[1,0].axes.set_xlabel('Number of Agents')
    axs[1,0].axes.set_ylabel('Time(min)')
    axs[1,0].set_title('Wall Time (CP)')
    # axs[0,1].plot(df_mean_CP['N'],df_mean_CP['Objective'])
    # axs[0,1].legend(['CP'])
    
    
    # Norm=((df_mean_DP['Objective_T']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
    Norm_mean_Sorted=((df_mean_DP_Sorted['Objective_T']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
    Norm_mean_Random=((df_mean_DP_Random['Objective_T']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100

    # Norm2=((df_mean_DP['Objective']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
    
    axs[0,1].plot(df_mean_DP['N'], Norm_mean_Sorted, color='k')
    axs[0,1].plot(df_mean_DP['N'], Norm_mean_Random, color='gold')
    
    axs[0,1].legend(['Sorted','Random'])
    # axs[0,1].plot(df_mean_DP['N'], Norm2)
    # axs[0,1].legend(['Dp'])
    axs[0,1].axes.set_xticks(df_DP['N'])
    axs[0,1].grid()
    axs[0,1].axes.set_xlabel('Number of Agents')
    axs[0,1].axes.set_ylabel('%')
    axs[0,1].set_title('DP objective (trans) relative to CP optimal')
    
    axs[1,1].plot(df_mean_CP['N'],df_mean_CP['SSR']*100, color='blue')
    axs[1,1].plot(df_mean_DP_Sorted['N'], df_mean_DP_Sorted['SSR']*100, color='k')
    axs[1,1].plot(df_mean_DP_Random['N'],df_mean_DP_Random['SSR']*100, color='gold')
    axs[1,1].legend(['CP-Optimal','Sorted','Random'])
    axs[1,1].axes.set_xticks(df_DP['N'])
    axs[1,1].grid()
    axs[1,1].axes.set_xlabel('Number of Agents')
    axs[1,1].axes.set_ylabel('%')
    axs[1,1].set_title('Self-Suficiency Ratio')
    
    axs[2,1].plot(df_mean_CP['N'],df_mean_CP['SCR']*100, color='blue')
    # axs[2,1].plot(df_mean_DP['N'],df_mean_DP['SCR'], color='k')
    axs[2,1].plot(df_mean_DP_Sorted['N'], df_mean_DP_Sorted['SCR']*100, color='k')
    axs[2,1].plot(df_mean_DP_Random['N'],df_mean_DP_Random['SCR']*100, color='gold')
    axs[2,1].legend(['CP-Optimal','Sorted','Random'])

    axs[2,1].axes.set_xticks(df_DP['N'])
    axs[2,1].grid()
    axs[2,1].axes.set_xlabel('Number of Agents')
    axs[2,1].axes.set_ylabel('%')
    axs[2,1].set_title('Self-Consumption Rate')
    
    plt.tight_layout()        
    plt.show()
    
            # assert df_CP['N'].equals(df_DP['N']), 'Number of agents diffreent in DP and CP'
        # %% Only DP
        # axs[0,0].plot(df_DP['N'],df_DP['Wall_Time']/60)
        # axs[0,0].axes.set_xticks(df_DP['N'])
        # # axs[0,0].legend(['Distributed Problem'])
        # axs[0,0].grid()
        # axs[0,0].axes.set_xlabel('Number of Agents')
        # axs[0,0].axes.set_ylabel('Time(min)')
        # axs[0,0].set_title('Wall Time (DP)')
    
        # axs[1,0].plot(df_CP['N'],df_CP['Wall_Time']/60)
        # axs[1,0].axes.set_xticks(df_CP['N'])
        # # axs[1,0].legend(label)
        # axs[1,0].grid()
        # axs[1,0].axes.set_xlabel('Number of Agents')
        # axs[1,0].axes.set_ylabel('Time(min)')
        # axs[1,0].set_title('Wall Time (CP)')
        
        # Norm_temp=((df_DP['Objective_T']-df_CP['Objective'])/df_CP['Objective'])*100
        # # Norm2=((df_mean_DP['Objective']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
        # axs[0,1].plot(df_DP['N'],Norm_temp)
        # # axs[0,1].plot(df_CP['N'],df_CP['Objective'])
        # axs[0,1].axes.set_xticks(df_DP['N'])
        # axs[0,1].grid()
        # axs[0,1].axes.set_xlabel('Number of Agents')
        # axs[0,1].axes.set_ylabel('€')
        # axs[0,1].set_title('Objective function')
        
        # # axs[1,1].plot(N,df_DP['SSR'].astype(float),color='red')
        # axs[1,1].plot(df_CP['N'],df_CP['SSR'])
        # axs[1,1].plot(df_DP['N'],df_DP['SSR'])
        # axs[1,1].axes.set_xticks(df_CP['N'])
        # axs[1,1].grid()
        # # axs[1,1].axes.set_xlabel('Number of Agents')
        # axs[1,1].axes.set_ylabel('%')
        # axs[1,1].set_title('Self-Suficiency Ratio')
        
        # # axs[2,0].plot(DevMean_temp['N'],DevMean_temp['m_p'])
        # axs[2,0].plot(DevMean_temp['N'],DevMean_temp['m_d'])
        # axs[2,0].axes.set_xticks(df_CP['N'])
        # # axs[2,0].legend(label)
        # axs[2,0].grid()
        # axs[2,0].axes.set_xlabel('Number of Agents')
        # axs[2,0].axes.set_ylabel('kW')
        
        
        # axs[2,1].plot(df_CP['N'],df_CP['SCR'])
        # axs[2,1].plot(df_DP['N'],df_DP['SCR'])
        # axs[2,1].axes.set_xticks(df_CP['N'])
        # axs[2,1].grid()
        # axs[2,1].axes.set_xlabel('Number of Agents')
        # axs[2,1].axes.set_ylabel('%')
        # axs[2,1].set_title('Self-Consumption Rate')

            # file=ResultsFolder + '/Compare'
            # plt.savefig(file,dpi=300)
    
    
    
    
    
#%% Simple fast plot for a specific N

def PlotCompareFixN(df,Folder):
    
    fig, axs = plt.subplots(2, 2)

    axs[0,0].plot(df['Model'],df['Wall_Time'].astype(float),color='red')
    axs[0,0].axes.set_xticks(df['Model'])
    axs[0,0].axes.set_xticklabels(df['Model'], Rotation=90)
    axs[0,0].grid()
    axs[0,0].axes.set_xlabel('Model')
    axs[0,0].axes.set_ylabel('Time(s)')
    axs[0,0].set_title('Wall Time')

    axs[1,0].plot(df['Model'],df['Objective'].astype(float))
    axs[1,0].axes.set_xticks(df['Model'])
    axs[1,0].legend(['Centralized Problem'],loc='upper left')
    axs[1,0].grid()
    axs[1,0].axes.set_xlabel('Model')
    axs[1,0].axes.set_ylabel('Obj')

    axs[0,1].plot(df['Model'],df['Tshift'].astype(float))
    axs[0,1].axes.set_xticks(df['Model'])
    axs[0,1].grid()
    axs[0,1].axes.set_xlabel('Model')
    axs[0,1].axes.set_ylabel('E')
    axs[0,1].set_title('Energy')

    axs[1,1].plot(N,df['SSR'].astype(float),color='red')
    axs[1,1].axes.set_xticks(N)
    axs[1,1].grid()
    axs[1,1].axes.set_xlabel('Number of Agents')
    axs[1,1].axes.set_ylabel('%')
    axs[1,1].set_title('Self-Suficiency Ratio')
    
    file=ResultsFolder + '/Compare'
    plt.savefig(file,dpi=300)
    plt.show()

#%% Voxels
def voxelsfunc(MatFile, d):

    Results=sio.loadmat(MatFile)

    M=Results['P']
    Pag=Results['P_ag']
    Ppv=Results['Ppv']

    x, y, z = np.indices((len(M[0]), len(M)+3, d))

    M=ResultsCP['P']
    Pag=ResultsCP['P_ag']
    Ppv=ResultsCP['Ppv']
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for k in range(len(M)):
        m=M[k,:]
        idx=np.nonzero(m)
        idx=idx[0]
        xi=idx[0]
        xf=idx[len(idx)-1]
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        # cube = (x >= xi) & (x <= xf) & (y < 1) & (z < max(m))
        cube = (x >= xi) & (x <= xf) & (y >= k+1) & (y < k+2) & (z <= max(m))
        ax.voxels(cube, facecolors=color, edgecolor='k')
    
    # plt.show()
    
    # for i in range(len(Pag[0])):
    #     p=Pag[0][i]
    #     # print(p)
    #     cubeag = (x >= i) & (x <= i+1) & (y >= n+2) & (z < p)
    #     ax.voxels(cubeag, facecolors='blue', edgecolor='k',label='parametric curve')
        
    for i in range(len(Ppv)):
        p=Ppv[i]
        print(p)
        cubeag = (x >= i) & (x <= i+1) & (y >= n+2) & (z < p)
        ax.voxels(cubeag, facecolors='gold', edgecolor='k',label='parametric curve')    
    
    ax.set_xlabel('Time of the day')
    ax.set_ylabel('Agents')
    ax.set_zlabel('Power (kW)')
    plt.show()


# %%Plots for Paper
def SimplePlot(x,y1,y2,H,fw,color, color2,s):
    
    if s=='Simple':
        #make figure
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('hour of the day', fontsize=fw)
        # ax1.set_ylabel('kW', color=color,fontsize=fw)
        ax1.set_ylabel('kW',fontsize=fw)
        ax1.plot(y1, color=color,linewidth=3)
        ax1.tick_params(axis='y',labelsize=fw)
        div=12
        L=np.linspace(0,H,div,endpoint=False)
        ax1.set_xticks(L)
        ax1.set_xticklabels(np.linspace(0,24,div,dtype=int,endpoint=False),fontsize=fw)
        ax1.grid()
        print('Simple Plot')
        
    elif s=='TwoAxis':
        
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Number of devices', fontsize=fw)
        ax1.set_ylabel('kW',fontsize=fw)
        ax1.plot(x,y1, color=color,linewidth=3)
        ax1.tick_params(axis='y',labelsize=fw)
        ax1.set_ylabel('PV Installed Capacity (kWp)',color=color,fontsize=fw)
        
        div=12
        
        ax1.set_xticks(x)
        # ax1.set_xticklabels(x,fontsize=fw)
        # ax1.grid()
        ax1.grid()
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis'
        ax2.set_ylabel('kW', color=color2)  # we already handled the x-label with ax1    
        ax2.plot(x,y2, color,linewidth=3.0) 
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.tick_params(axis='y', labelsize=fw)
        ax2.set_xticklabels(x,fontsize=fw)
        ax2.set_ylabel('Devices Total Demand (kWh)',color=color,fontsize=fw)
        
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        # file=ResultsFolder + '/CP_N_%i' %n + RunFile
        # plt.savefig(file,dpi=300)
        print('2 Axis')
    
    
    plt.show()
