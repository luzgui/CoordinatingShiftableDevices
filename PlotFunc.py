from __future__ import division
from pyomo.environ import *
from pyomo.opt import SolverFactory
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import matplotlib



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
def PlotCompare(df,ResultsFolder, Appsfiles,DevMeanFile,stat,a,linebig,fw):
    
    'df: data frame with the results'
    'Resultsfolder: images destination'
    'Appsfiles: list of Appsfiles names'
    'DevMeanFile: a path to a CSV with the DevicesList_Mean dataframe (check mainscript) '
    
    DevicesList_Mean=pd.read_csv(DevMeanFile)
    
    df['Wall_Time']=df['Wall_Time'].astype(float)
    
    df_mean_CP = pd.DataFrame(columns=df.columns) #for mean calc
    df_min_CP = pd.DataFrame(columns=df.columns) #for min calc
    df_max_CP = pd.DataFrame(columns=df.columns) #for min calc
    
    df_mean_DP = pd.DataFrame(columns=df.columns)
    df_mean_DP_Sorted = pd.DataFrame(columns=df.columns)
    df_min_DP_Sorted = pd.DataFrame(columns=df.columns) #for min calc
    df_max_DP_Sorted = pd.DataFrame(columns=df.columns) #for min cal
    
    df_mean_DP_Random = pd.DataFrame(columns=df.columns)
    df_min_DP_Random = pd.DataFrame(columns=df.columns) #for min calc
    df_max_DP_Random = pd.DataFrame(columns=df.columns) #for min ca
    
    # Calculate the means for each N
    
    df=df.loc[df['AppsList'].isin(Appsfiles)] # its important if there are other mat files in folder
    df=df[df.N !=15]
    df_CP=df[df.Model.str.contains('CP', case=True)]
    df_DP=df[df.Model.str.contains('DP', case=True)]
    
    df_DP_Sorted=df[df.Model.str.contains('DP_Sorted', case=True)]
    df_DP_Random=df[df.Model.str.contains('DP_Random', case=True)]
    
    SD=pd.DataFrame(columns=['N','CP','DP_S','DP_R'])
    
    for i in df.N.unique():
        df_temp_CP=df_CP.loc[df_CP.N==i]
        df_temp_DP=df_DP.loc[df_DP.N==i]
        df_temp_DP_Sorted=df_DP_Sorted.loc[df_DP_Sorted.N==i]
        df_temp_DP_Random=df_DP_Random.loc[df_DP_Random.N==i]
        # SD['N']=i
        # SD['CP']=df_temp_CP.Wall_Time.std()
        # print(df_temp)
        #Incredible sequance of 3 functions applied
        if stat=='Mean':
            df_mean_CP=pd.concat([df_mean_CP,df_temp_CP.mean().to_frame().transpose()], sort=True)
            df_mean_DP=pd.concat([df_mean_DP,df_temp_DP.mean().to_frame().transpose()], sort=True)
            df_mean_DP_Sorted=pd.concat([df_mean_DP_Sorted,df_temp_DP_Sorted.mean().to_frame().transpose()], sort=True)
            df_mean_DP_Random=pd.concat([df_mean_DP_Random,df_temp_DP_Random.mean().to_frame().transpose()], sort=True)
            MeanType='Mean'
        #median
        elif stat=='Median':
            df_mean_CP=pd.concat([df_mean_CP,df_temp_CP.median().to_frame().transpose()],sort=True)
            df_mean_DP=pd.concat([df_mean_DP,df_temp_DP.median().to_frame().transpose()],sort=True)
            df_mean_DP_Sorted=pd.concat([df_mean_DP_Sorted,df_temp_DP_Sorted.median().to_frame().transpose()],sort=True)
            df_mean_DP_Random=pd.concat([df_mean_DP_Random,df_temp_DP_Random.median().to_frame().transpose()],sort=True)
        
            MeanType='Median'
        
        #Get the Minimum Times
        
        df_temp_CP_min=df_temp_CP.loc[df_temp_CP.Wall_Time==df_temp_CP.Wall_Time.min()]
        df_temp_CP_max=df_temp_CP.loc[df_temp_CP.Wall_Time==df_temp_CP.Wall_Time.max()]
        df_min_CP=pd.concat([df_min_CP,df_temp_CP_min], sort=True)
        df_max_CP=pd.concat([df_max_CP,df_temp_CP_max], sort=True)
    
        df_temp_DP_Sorted_min=df_temp_DP_Sorted.loc[df_temp_DP_Sorted.Wall_Time==df_temp_DP_Sorted.Wall_Time.min()]
        df_temp_DP_Sorted_max=df_temp_DP_Sorted.loc[df_temp_DP_Sorted.Wall_Time==df_temp_DP_Sorted.Wall_Time.max()]
        df_min_DP_Sorted=pd.concat([df_min_DP_Sorted,df_temp_DP_Sorted_min], sort=True)
        df_max_DP_Sorted=pd.concat([df_max_DP_Sorted,df_temp_DP_Sorted_max], sort=True)
        
        df_temp_DP_Random_min=df_temp_DP_Random.loc[df_temp_DP_Random.Wall_Time==df_temp_DP_Random.Wall_Time.min()]
        df_temp_DP_Random_max=df_temp_DP_Random.loc[df_temp_DP_Random.Wall_Time==df_temp_DP_Random.Wall_Time.max()]
        df_min_DP_Random=pd.concat([df_min_DP_Random,df_temp_DP_Random_min], sort=True)
        df_max_DP_Random=pd.concat([df_max_DP_Random,df_temp_DP_Random_max], sort=True)
        
    df_min_CP=df_min_CP.sort_values(by='N')
    df_max_CP=df_max_CP.sort_values(by='N')
    df_min_DP_Sorted=df_min_DP_Sorted.sort_values(by='N') 
    df_max_DP_Sorted=df_max_DP_Sorted.sort_values(by='N') 
    df_min_DP_Random=df_min_DP_Random.sort_values(by='N') 
    df_max_DP_Random=df_max_DP_Random.sort_values(by='N') 
    
    df_mean_CP=df_mean_CP.sort_values(by='N')    
    df_mean_DP=df_mean_DP.sort_values(by='N')   
    df_mean_DP_Sorted=df_mean_DP_Sorted.sort_values(by='N')   
    df_mean_DP_Random=df_mean_DP_Random.sort_values(by='N')   
    
    #Common X axis (Number of agents)
    N=df_mean_CP['N'].astype(int)
    
    # Atributes for plots
    colorRuns = plt.get_cmap('Pastel2')
    colorMean = plt.get_cmap('tab20b')

    #%% Figure
    import seaborn as sns
    plt.style.use('seaborn-darkgrid')
    figcp, axscp = plt.subplots(3, 1)
    plt.style.use('seaborn-darkgrid')
    figdp, axsdp = plt.subplots(2, 2)
    
    TitleTime='Computational Time'
    ylabelTime='Time(minutes)'
    xlabelAgents='Number of Agents'
    
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
        axsdp[0,0].plot(df_DP_Sorted['N'],df_DP_Sorted['Wall_Time']/60, color=colorMean(1), alpha=a)
        axsdp[0,0].plot(df_DP_Random['N'],df_DP_Random['Wall_Time']/60, color=colorMean(14), alpha=a)

        axsdp[0,0].axes.set_xticks(df_DP['N'])
        # axsdp[0,0].legend(['Sorted','Random'])
        # axsdp[0,0].grid()
        # axsdp[0].axes.set_xlabel('Number of Agents')
        axsdp[0,0].axes.set_ylabel(ylabelTime)
        # axsdp[0].set_title('Wall Time (DP)')
        
        Norm_temp=((df_DP['Objective_T']-df_CP['Objective'])/df_CP['Objective'])*100
        Norm_temp_Sorted=((df_DP_Sorted['Objective_T']-df_CP['Objective'])/df_CP['Objective'])*100
        Norm_temp_Random=((df_DP_Random['Objective_T']-df_CP['Objective'])/df_CP['Objective'])*100
        # print(Norm_temp_Sorted)
        # Norm2=((df_mean_DP['Objective']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
        axsdp[0,1].plot(df_DP_Sorted['N'],Norm_temp_Sorted,color=colorMean(1), alpha=a)
        axsdp[0,1].plot(df_DP_Random['N'],Norm_temp_Random,color=colorMean(14), alpha=a)
        # axsdp[0,1].legend(['Sorted','Random'])
        # axsdp[0,1].plot(df_CP['N'],df_CP['Objective'])
        axsdp[0,1].axes.set_xticks(df_DP['N'])
        # axsdp[0,1].grid()
        # axsdp[1].axes.set_xlabel('Number of Agents')
        axsdp[0,1].axes.set_ylabel('%')
        # axsdp[1].set_title('DP objective (trans) relative to CP optimal')
        
        # axsdp[1,1].plot(N,df_DP['SSR'].astype(float),color='red')
        axsdp[1,0].plot(df_CP['N'],df_CP['SSR']*100,color='k' )
        axsdp[1,0].plot(df_DP_Sorted['N'],df_DP_Sorted['SSR']*100, color=colorMean(1), alpha=a)
        axsdp[1,0].plot(df_DP_Random['N'],df_DP_Random['SSR']*100, color=colorMean(14), alpha=a)
        # axsdp[1,1].legend(['CP-Optimal','Sorted','Random'])
        axsdp[1,0].axes.set_xticks(N)
        # axsdp[1,1].grid()
        # axsdp[1,1].axes.set_xlabel('Number of Agents')
        axsdp[1,0].axes.set_ylabel('%')
        # axsdp[2].set_title('Self-Suficiency Ratio')
        
        # if k==Appsfiles[0]:
        #     axsdp2=axsdp[2].twinx()
        
        # axsdp2.plot(df_CP['N'],df_CP['SCR']*100,color='k')
        # # axs1[2,1].plot(df_DP['N'],df_DP['SCR'])
        # axsdp2.plot(df_DP_Sorted['N'],df_DP_Sorted['SCR']*100, color=colorMean(1), alpha=a)
        # axs1[2,1].plot(df_DP_Random['N'],df_DP_Random['SCR']*100, color=colorMean(14), alpha=a)
        # axs1[2,1].legend(['CP-Optimal','Sorted','Random'])
        # axs1[2,1].axes.set_xticks(df_CP['N'])
        # axs1[2,1].axes.set_xlabel('Number of Agents')
        # axs1[2,1].axes.set_ylabel('%')
        # axs1[2,1].set_title('Self-Consumption Rate')
        # axs1[2,1].grid()
        
        # axs1[2,0].plot(DevMean_temp['N'],DevMean_temp['m_p'])
        # axs1[2,0].plot(DevMean_temp['N'],DevMean_temp['m_d'])
        # axs1[2,0].axes.set_xticks(df_CP['N'])
        # # axs1[2,0].legend(label)
        # axs1[2,0].grid()
        # axs1[2,0].axes.set_xlabel('Number of Agents')
        # axs1[2,0].axes.set_ylabel('kW')
        
        
        
        
        #CP pLot
        
        axscp[0].plot(df_CP['N'],df_CP['Wall_Time']/60,color='k', alpha=a)
        # axscp[0].axes.set_xticks(df_CP['N'])
        # axscp[0].axes.set_xlabel('Number of Agents')
        # axscp[0].axes.set_ylabel('Time(min)')
        # axscp[0].set_title('Wall Time (CP)')
    
        axscp[1].plot(df_CP['N'],df_CP['Objective'],color='k', alpha=a)
        # axscp[1].axes.set_xticks(df_CP['N'])
        # axscp[1].axes.set_xlabel('Number of Agents')
        # axscp[1].axes.set_ylabel('€')
        # axscp[1].set_title('Objective')

    
    axsdp[0,0].axes.set_xticks(df_mean_CP['N'])
    axsdp[0,1].axes.set_xticks(df_mean_CP['N'])
    axsdp[1,0].axes.set_xticks(df_mean_CP['N'])
    
    axsdp[0,0].grid()
    axsdp[0,1].grid()
    axsdp[1,0].grid()
    # plt.tight_layout()        
    plt.subplots_adjust(top=0.984, bottom=0.056, left=0.524, right=0.992, hspace=0.124, wspace=0.2)
    
    

    
    
    
    plt.show()
            
    # fig, axs1 = plt.subplots(3, 2)
    axsdp[0,0].plot(df_mean_DP_Sorted['N'],df_mean_DP_Sorted['Wall_Time']/60, color=colorMean(1),linewidth=linebig,label='Sorted-' + MeanType)
    axsdp[0,0].plot(df_mean_DP_Random['N'],df_mean_DP_Random['Wall_Time']/60,color=colorMean(14),linewidth=linebig,label='Random-'+ MeanType)
    
    # axsdp[0,0].plot(df_mean_DP_Sorted['N'],df_mean_DP_Sorted['Wall_Time']/60,color=colorRuns(1),linewidth=linebig)
    # axsdp[0,0].plot(df_mean_DP_Random['N'],df_mean_DP_Random['Wall_Time']/60,color=colorRuns(2),linewidth=linebig)
    # axsdp[0,0].legend(['Sorted-' + MeanType,'Random-'+ MeanType], mode='expand',ncol=2, borderaxespad=0,bbox_to_anchor=(0., 1.02, 1., .102), loc='upper left')
        # axsdp[0,0].legend(mode='expand',ncol=2,bbox_to_anchor=(0., 1.02, 1., .102), loc='upper left')
    axsdp[0,0].legend()
    # axsdp[0].axes.set_xticks(df_mean_DP['N'])
    axsdp[0,0].grid()
    # axsdp[0,0].axes.set_xlabel('Number of Agents')
    # axsdp[0,0].axes.set_ylabel('Time(min)')
    # axsdp[0,0].set_title('Wall Time (DP)')
    
    axscp[0].plot(df_mean_CP['N'],df_mean_CP['Wall_Time']/60,color='k', linewidth=linebig, label='CP ' + MeanType)
    axscp[0].legend()
    axscp[0].axes.set_xticks(N)
    axscp[0].grid()
    axscp[0].axes.set_xlabel(xlabelAgents)
    axscp[0].axes.set_ylabel(ylabelTime)
    axscp[0].set_title(TitleTime + ' CP')
    
    axscp[1].plot(df_mean_CP['N'],df_mean_CP['Objective'],color='k',linewidth=linebig, label='CP ' + MeanType)
    axscp[1].legend()
    axscp[1].axes.set_xticks(N)
    axscp[1].grid()
    axscp[1].axes.set_xlabel(xlabelAgents)
    axscp[1].axes.set_ylabel('€')
    axscp[1].set_title('Objective')
    # axscp[0,1].plot(df_mean_CP['N'],df_mean_CP['Objective'])
    # axscp[0,1].legend(['CP'])
    
    
    # Norm=((df_mean_DP['Objective_T']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
    Norm_mean_Sorted=((df_mean_DP_Sorted['Objective_T']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
    Norm_mean_Random=((df_mean_DP_Random['Objective_T']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100

    # Norm2=((df_mean_DP['Objective']-df_mean_CP['Objective'])/df_mean_CP['Objective'])*100
    
    axsdp[0,1].plot(df_mean_DP['N'], Norm_mean_Sorted, color=colorMean(1),linewidth=linebig,label='Sorted-' + MeanType)
    axsdp[0,1].plot(df_mean_DP['N'], Norm_mean_Random, color=colorMean(14),linewidth=linebig,label='Random-' + MeanType)
    
    axsdp[0,1].legend()
    # axsdp[0,1].plot(df_mean_DP['N'], Norm2)
    # axsdp[0,1].legend(['Dp'])
    # axsdp[1].axes.set_xticks(df_DP['N'])
    axsdp[0,1].grid()
    # axsdp[1].axes.set_xlabel('Number of Agents')
    axsdp[0,1].axes.set_ylabel('%')
    # axsdp[1].set_title('DP objective (trans) relative to CP optimal')
    
    axsdp[1,0].plot(df_mean_CP['N'],df_mean_CP['SSR']*100, color='k', label='CP -Optimal')
    axsdp[1,0].plot(df_mean_DP_Sorted['N'], df_mean_DP_Sorted['SSR']*100, color=colorMean(1),linewidth=linebig,label='Sorted-' + MeanType)
    axsdp[1,0].plot(df_mean_DP_Random['N'],df_mean_DP_Random['SSR']*100, color=colorMean(14),linewidth=linebig,label='Random-' + MeanType)
    axsdp[1,0].legend()
    axsdp[1,0].axes.set_xticks(N)
    # axsdp[2].grid()
    axsdp[1,0].axes.set_xlabel(xlabelAgents)
    axsdp[1,0].axes.set_ylabel('%')
    # axsdp[2].set_title('Self-Suficiency Ratio')
    
    # axsdp2.plot(df_mean_CP['N'],df_mean_CP['SCR']*100, color='k', label='CP-Optimal')
    # # axsdp[2,1].plot(df_mean_DP['N'],df_mean_DP['SCR'], color='k')
    # axsdp[2,1].plot(df_mean_DP_Sorted['N'], df_mean_DP_Sorted['SCR']*100, color=colorMean(1),linewidth=linebig,label='Sorted-' + MeanType)
    # axsdp[2,1].plot(df_mean_DP_Random['N'],df_mean_DP_Random['SCR']*100, color=colorMean(14),linewidth=linebig,label='Random-' + MeanType)
    # axsdp[2,1].legend()

    # axsdp[2,1].axes.set_xticks(df_DP['N'])
    # axsdp[2,1].grid()
    # axsdp[2,1].axes.set_xlabel('Number of Agents')
    # axsdp[2,1].axes.set_ylabel('%')
    # axsdp[2,1].set_title('Self-Consumption Rate')
    
    plt.tight_layout()        
    plt.show()
    
    
    #Fitting Plots
    from sklearn.metrics import r2_score
    
    poly_CP_min=np.poly1d(np.polyfit(N.values, df_min_CP['Wall_Time'].values, 1))
    poly_DP_Sorted_min=np.poly1d(np.polyfit(N.values, df_min_DP_Sorted['Wall_Time'].values, 1))
    poly_DP_Random_min=np.poly1d(np.polyfit(N.values, df_min_DP_Random['Wall_Time'].values, 1))
    
    poly_CP_max=np.poly1d(np.polyfit(N.values, df_max_CP['Wall_Time'].values, 1))
    poly_DP_Sorted_max=np.poly1d(np.polyfit(N.values, df_max_DP_Sorted['Wall_Time'].values, 1))
    poly_DP_Random_max=np.poly1d(np.polyfit(N.values, df_max_DP_Random['Wall_Time'].values, 1))
    
    poly_CP_mean=np.poly1d(np.polyfit(N.values, df_mean_CP['Wall_Time'].values, 1))
    poly_DP_Sorted_mean=np.poly1d(np.polyfit(N.values, df_mean_DP_Sorted['Wall_Time'].values, 1))
    poly_DP_Random_mean=np.poly1d(np.polyfit(N.values, df_mean_DP_Random['Wall_Time'].values, 1))
    
    #R2 values
    R2_Mean_CP = r'$R^2=%.2f$' % (r2_score(df_mean_CP['Wall_Time'], poly_CP_mean(N)),)
    R2_Mean_Sort = r'$R^2=%.2f$' % (r2_score(df_mean_DP_Sorted['Wall_Time'], poly_DP_Sorted_mean(N)), )
    R2_Mean_Random = r'$R^2=%.2f$' % (r2_score(df_mean_DP_Random['Wall_Time'], poly_DP_Random_mean(N)), )
    
    R2_Min_CP = r'$R^2=%.2f$' % (r2_score(df_min_CP['Wall_Time'], poly_CP_min(N)),)
    R2_Min_Sort = r'$R^2=%.2f$' % (r2_score(df_min_DP_Sorted['Wall_Time'], poly_DP_Sorted_min(N)), )
    R2_Min_Random = r'$R^2=%.2f$' % (r2_score(df_min_DP_Random['Wall_Time'], poly_DP_Random_min(N)), )
    
    #Plots
    
    plt.style.use('seaborn-darkgrid')
    fig2, axs2 = plt.subplots(2, 2)
    
    axs2[0,0].plot(N,df_min_CP['Wall_Time']/60,label='CP-min')
    axs2[0,0].plot(N, poly_CP_min(N)/60, label='CP-poly-min '+ ( R2_Min_CP))
        
    # r2_score(df_min_CP['Wall_Time'], poly_CP_min(N))
    
    # axs2[0,0].plot(N,df_max_CP['Wall_Time']/60,label='CP-max')
    # axs2[0,0].plot(N, poly_CP_max(N)/60, label='CP-poly-max')
    
    axs2[0,0].legend()
    axs2[0,0].axes.set_xticks(N)
    axs2[0,0].grid()
    # axs2[0,0].axes.set_xlabel(xlabelAgents)
    axs2[0,0].axes.set_ylabel('min')
    axs2[0,0].set_title(TitleTime)
    
        

    axs2[0,1].plot(N,df_mean_CP['Wall_Time']/60,label='CP-mean')
    axs2[0,1].plot(N, poly_CP_mean(N)/60, label='CP-poly-mean '+ ( R2_Mean_CP))
    # r2_score(df_mean_CP['Wall_Time'], poly_CP_mean(N))
    axs2[0,1].legend()
    axs2[0,1].axes.set_xticks(N)
    axs2[0,1].grid()
    # axs2[0,1].axes.set_xlabel(xlab)
    # axs2[0,1].axes.set_ylabel('min')
    axs2[0,1].set_title('Walltime mins')


    
    axs2[1,1].plot(N,df_mean_DP_Sorted['Wall_Time']/60,label='DP-Sorted-mean')
    axs2[1,1].plot(N,df_mean_DP_Random['Wall_Time']/60,label='DP-Random-mean')
    axs2[1,1].plot(N, poly_DP_Sorted_mean(N)/60, label='DP-Sorted-poly-mean '+ (R2_Mean_Sort))
    axs2[1,1].plot(N, poly_DP_Random_mean(N)/60, label='DP-Random-poly-mean '+ (R2_Mean_Random))
    

    axs2[1,1].legend()
    axs2[1,1].axes.set_xticks(N)
    axs2[1,1].grid()
    axs2[1,1].axes.set_xlabel(xlabelAgents)
    # axs2[1,1].axes.set_ylabel(ylabelTime)
    # axs2[1,1].set_title(TitleTime)
     

    axs2[1,0].plot(N,df_min_DP_Sorted['Wall_Time']/60,label='Sorted-min')
    axs2[1,0].plot(N,poly_DP_Sorted_min(N)/60,label='Sorted-poly-min '+ ( R2_Min_Sort))
    axs2[1,0].plot(N,df_min_DP_Random['Wall_Time']/60,label='Random-min')
    axs2[1,0].plot(N,poly_DP_Random_min(N)/60,label='Random-poly-min '+ ( R2_Min_Random))
    
    axs2[1,0].legend()
    axs2[1,0].axes.set_xticks(N)
    
    axs2[1,0].axes.set_xlabel(xlabelAgents)
    axs2[1,0].axes.set_ylabel(ylabelTime)
    # axs2[1,0].set_title('Walltime mins')
    axs2[1,0].grid()
    # axs2[1,0].plot(N,df_max_DP_Sorted['Wall_Time']/60,label='DP-Sorted-max')
    # axs2[1,0].plot(N, poly_DP_Sorted_max(N)/60, label='DP-Sorted-poly-max')
    
    plt.tight_layout()        
    plt.show()
    
            # assert df_CP['N'].equals(df_DP['N']), 'Number of agents diffreent in DP and CP'


            # file=ResultsFolder + '/Compare'
            # plt.savefig(file,dpi=300)

    return df_mean_CP, df_mean_DP_Random, df_mean_DP_Sorted, df_min_CP, df_min_DP_Random, df_min_DP_Sorted, 
    





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
def voxelsfunc(MatFile, d):
#%% Voxels


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





def ProfilePlot(Matfiles,TarInit, fw,lw, color,color2):

    import seaborn as sns
    import scipy.io as sio
    # plt.style.use('seaborn-darkgrid')
    figcp, axscp = plt.subplots(1, len(Matfiles))
    
    xlabel='hour of the day'
    ylabel='kW'
    y2label='€'
        
    m=-1
    for k in Matfiles:
        m += 1
        print(m)

        Results=sio.loadmat(k)
        P=Results['P']

        color_list = plt.cm.Set2(np.linspace(0, 1, len(P)))
        # color_list = plt.cm.Dark2(np.linspace(0, 1, len(P)))
        
        P=pd.DataFrame(P)
        Pcum=P.cumsum()
        Pag=Results['P_ag']
        Ppv=Results['Ppv']
        
        axscp[m].plot(Pag[0], color='k',linewidth=lw, label='Aggregated demand')
        axscp[m].plot(Ppv, color=color2,linewidth=lw, label='PV profile')
        for i in range(P.shape[0]):
            axscp[m].plot(Pcum.iloc[Pcum.shape[0]-1-i,:], color='k')     
            axscp[m].fill(Pcum.iloc[Pcum.shape[0]-1-i,:],color=color_list[i])
            axscp[m].set_xlabel(xlabel, fontsize=fw)
            axscp[m].set_ylabel(ylabel, color=color,fontsize=fw)
            axscp[m].tick_params(axis='y',labelsize=fw)
            
            div=12
            L=np.linspace(0,144,div,endpoint=False)
            axscp[m].set_xticks(L)
            axscp[m].set_xticklabels(np.linspace(0,24,div,dtype=int,endpoint=False),fontsize=fw)
            axscp[m].set_title('%i Agents' %len(P) + Results['Model'][0])
            axscp[m].legend(loc='center right')
        axscp2=axscp[m].twinx()  # instantiate a second axes that shares the same x-axis'
        TarLabel='PV indexed tariff'
        if 'CP' in k:
            axscp2.plot(TarInit, color='k',linewidth=lw, label=TarLabel)    
        elif 'DP' in k:
            axscp2.plot(Results['Tar'][len(P)-1], color='k', linewidth=lw,label=TarLabel)  
            
        axscp2.set_ylabel(y2label, color=color,fontsize=fw)  # we already handled the x-label with ax1
        axscp2.tick_params(axis='y', labelsize=fw)
        
        # figcp.tight_layout()  # otherwise the right y-label is slightly clipped
        # file=ResultsFolder + 'ss'
        # plt.savefig(file,dpi=300)
        plt.show()
    
    
