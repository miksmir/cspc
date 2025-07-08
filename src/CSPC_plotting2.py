# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 11:20:17 2025

@author: Mikhail
"""

"""
This script takes the reconstruction values from arrays after point cloud 
reconstruction and plots them in terms of sparsity, basis chosen, and
compression ratio. These are all plotted as 2D scatter plots and bar graphs.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from CSPointCloud import OUTPUT_PATH_PLOTS
from CSPC_compilecsv import extract_csv

# Notes: Coif1 seems to work slightly better for Mt Marcy
# But Db1 seems to work slightly better for Columbus Circle



sparsities, MtMarcyMAE_dct = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, MtMarcyMAE_haar = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Haar DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, MtMarcyMAE_db2 = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, MtMarcyMAE_coif1 = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Coif1 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')

__, DeerparkMAE_dct = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, DeerparkMAE_haar = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Haar DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, DeerparkMAE_db2 = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, DeerparkMAE_coif1 = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Coif1 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')

__, ColCircMAE_dct = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ColCircMAE_haar = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="Haar DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ColCircMAE_db2 = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ColCircMAE_coif1 = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="Coif1 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')

__, ParkslopeMAE_dct = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ParkslopeMAE_haar = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="Haar DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ParkslopeMAE_db2 = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ParkslopeMAE_coif1 = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="Coif1 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
""" --- """
cs_ratios, MtMarcyMAE_dct_s = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="DCT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
cs_ratios_decimal = 0.01 * cs_ratios
__, MtMarcyMAE_haar_s = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Haar DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, MtMarcyMAE_db2_s = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Db2 DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, MtMarcyMAE_coif1_s = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Coif1 DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')

__, DeerparkMAE_dct_s = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="DCT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, DeerparkMAE_haar_s = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Haar DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, DeerparkMAE_db2_s = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Db2 DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, DeerparkMAE_coif1_s = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Coif1 DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')

__, ColCircMAE_dct_s = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="DCT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, ColCircMAE_haar_s = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="Haar DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, ColCircMAE_db2_s = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="Db2 DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, ColCircMAE_coif1_s = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="Coif1 DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')

__, ParkslopeMAE_dct_s = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="DCT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, ParkslopeMAE_haar_s = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="Haar DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, ParkslopeMAE_db2_s = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="Db2 DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
__, ParkslopeMAE_coif1_s = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="Coif1 DWT" and Sparsity==90 and `CS Ratio`>=25', x_axis='CS Ratio', y_axis='Mean Absolute Error')
""" --- """
sparsities, MtMarcyMAE_cs25 = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, MtMarcyMAE_cs50 = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==50', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, MtMarcyMAE_cs75 = extract_csv(csv_name='MtMarcy_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==75', x_axis='Sparsity', y_axis='Mean Absolute Error')

__, DeerparkMAE_cs25 = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, DeerparkMAE_cs50 = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==50', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, DeerparkMAE_cs75 = extract_csv(csv_name='Deerpark_5000points.csv', pd_query='Basis=="Db2 DWT" and `CS Ratio`==75', x_axis='Sparsity', y_axis='Mean Absolute Error')

__, ColCircMAE_cs25 = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ColCircMAE_cs50 = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==50', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ColCircMAE_cs75 = extract_csv(csv_name='ColumbusCircle_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==75', x_axis='Sparsity', y_axis='Mean Absolute Error')

__, ParkslopeMAE_cs25 = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==25', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ParkslopeMAE_cs50 = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==50', x_axis='Sparsity', y_axis='Mean Absolute Error')
__, ParkslopeMAE_cs75 = extract_csv(csv_name='Parkslope_5000points.csv', pd_query='Basis=="DCT" and `CS Ratio`==75', x_axis='Sparsity', y_axis='Mean Absolute Error')



    # Plots Reconstruction Error (y) over Compression (x), at different Bases (colorbar) as a line graph.
    # Same sparsity, different cs ratios, different basis
def plot_cs_basis(y1, y2, y3, y4, normalized:bool=True, xlabel='testx', ylabel='testy', title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='fig.pdf', bbox=(0,0,1,1), ylim=None, cspercentage=False):
    
    fig, ax = plt.subplots(figsize=sizefig)
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
        y4 = (y4 - y4.min()) / (y4.max()-y4.min())
    
    if(cspercentage==False):
        
        ax.plot(cs_ratios_decimal, y1, linestyle='-', marker='o', label='DCT', linewidth=3, markeredgewidth=2)
        ax.plot(cs_ratios_decimal, y2, linestyle='-', marker='o', label='Haar DWT', linewidth=3, markeredgewidth=2)
        ax.plot(cs_ratios_decimal, y3, linestyle='-', marker='o', label='Db2 DWT', linewidth=3, markeredgewidth=2)
        ax.plot(cs_ratios_decimal, y4, linestyle='-', marker='o', label='Coif1 DWT', linewidth=3, markeredgewidth=2)
        ax.legend(loc=loclegend, title='Basis', fontsize=labelsize, title_fontsize=labelsize+1, bbox_to_anchor=bbox)
        #ax2 = ax.twinx()
        #plt.rcParams['axes.labelsize'] = 20
        ax.tick_params(axis='both',labelsize=labelsize)
        ax.set_xlabel('Compression Ratio', fontsize=labelsize)
        ax.set_ylabel('MAE', fontsize=labelsize)
        ax.set_ylim(ylim)
        if(title != ''):
            ax.set_title(title, size=labelsize)
        ax.set_xticks([0.10, 0.25, 0.50, 0.75])
    else:
        #ax.scatter(x,y, marker='o')
        ax.plot(cs_ratios, y1, linestyle='-', marker='o', label='DCT', linewidth=3, markeredgewidth=2)
        ax.plot(cs_ratios, y2, linestyle='-', marker='o', label='Haar DWT', linewidth=3, markeredgewidth=2)
        ax.plot(cs_ratios, y3, linestyle='-', marker='o', label='Db2 DWT', linewidth=3, markeredgewidth=2)
        ax.plot(cs_ratios, y4, linestyle='-', marker='o', label='Coif1 DWT', linewidth=3, markeredgewidth=2)
        ax.legend(loc=loclegend, title='Basis', fontsize=labelsize, title_fontsize=labelsize+1, bbox_to_anchor=bbox)
        #ax2 = ax.twinx()
        #plt.rcParams['axes.labelsize'] = 20
        ax.tick_params(axis='both',labelsize=labelsize)
        ax.set_xlabel('Compression Ratio [%]', fontsize=labelsize)
        ax.set_ylabel('MAE', fontsize=labelsize)
        ax.set_ylim(ylim)
        if(title != ''):
            ax.set_title(title, size=labelsize)
        ax.set_xticks([25, 50, 75])
        
    plt.savefig(os.path.join(OUTPUT_PATH_PLOTS, figname), format='pdf', bbox_inches='tight')
    plt.show()
    
    # Plots Reconstruction Error (y) over Sparsity (x), at different Bases (colorbar) as a line graph.
    # Same cs ratio, different sparsity, different basis
def plot_basis_sparsity(y1, y2, y3, y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='fig.pdf', bbox=(0,0,1,1), ylim=None):
    fig, ax = plt.subplots(figsize=sizefig)
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
        y4 = (y4 - y4.min()) / (y4.max()-y4.min())
    
    # Choose for which CS Ratio to plot here!
    
    #ax.scatter(x,y, marker='o')
    ax.plot(sparsities, y1, linestyle='-', marker='o', label='DCT', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y2, linestyle='-', marker='o', label='Haar DWT', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y3, linestyle='-', marker='o', label='Db2 DWT', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y4, linestyle='-', marker='o', label='Coif1 DWT', linewidth=3, markeredgewidth=2)
    ax.legend(loc=loclegend, title='Basis', fontsize=labelsize+4, title_fontsize=labelsize+4, bbox_to_anchor=bbox)  
    #ax2 = ax.twinx()
    #plt.rcParams['axes.labelsize'] = 20
    ax.tick_params(axis='both',labelsize=labelsize+4)
    ax.set_xlabel('Sparsity [%]', fontsize=labelsize+3, labelpad=6)
    ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
    ax.set_ylim(ylim)
    '''
    if(title != ''):
        ax.set_title(title, size=labelsize)
    '''
    plt.savefig(os.path.join(OUTPUT_PATH_PLOTS, figname), format='pdf', bbox_inches='tight')
    plt.show()
    
    # Plots Reconstruction Error (y) over Sparsity (x), at different Compression Ratios (colorbar) as a line graph.
    # Same basis, different sparsity, different cs ratios
def plot_cs_sparsity(y1, y2, y3, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='fig.pdf', bbox=(0,0,1,1), ylim=None, cspercentage=False):
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
    
    fig, ax = plt.subplots(figsize=sizefig)
    
    if(cspercentage==False):
        #ax.scatter(x,y, marker='o')
        ax.plot(sparsities, y1, linestyle='-', marker='o', label='0.25', linewidth=3, markeredgewidth=2)
        ax.plot(sparsities, y2, linestyle='-', marker='o', label='0.50', linewidth=3, markeredgewidth=2)
        ax.plot(sparsities, y3, linestyle='-', marker='o', label='0.75', linewidth=3, markeredgewidth=2)
        ax.legend(loc=loclegend, title='Compression Ratio', fontsize=labelsize+3, title_fontsize=labelsize+4, bbox_to_anchor=bbox)  
        #ax2 = ax.twinx() 
        #plt.rcParams['axes.labelsize'] = 20
        ax.tick_params(axis='both',labelsize=labelsize+4)
        ax.set_xlabel('Sparsity [%]', fontsize=labelsize+4, labelpad=6)
        ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
        ax.set_ylim(ylim)
    else:
        #ax.scatter(x,y, marker='o')
        ax.plot(sparsities, y1, linestyle='-', marker='o', label='25%', linewidth=3, markeredgewidth=2)
        ax.plot(sparsities, y2, linestyle='-', marker='o', label='50%', linewidth=3, markeredgewidth=2)
        ax.plot(sparsities, y3, linestyle='-', marker='o', label='75%', linewidth=3, markeredgewidth=2)
        ax.legend(loc=loclegend, title='Compression Ratio', fontsize=labelsize+3, title_fontsize=labelsize+4, bbox_to_anchor=bbox)  
        #ax2 = ax.twinx() 
        #plt.rcParams['axes.labelsize'] = 20
        ax.tick_params(axis='both',labelsize=labelsize+4)
        ax.set_xlabel('Sparsity [%]', fontsize=labelsize+4, labelpad=6)
        ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
        ax.set_ylim(ylim)
    '''
    if(title != ''):
        ax.set_title(title, size=labelsize)
    '''
    plt.savefig(os.path.join(OUTPUT_PATH_PLOTS, figname), format='pdf', bbox_inches='tight')
    plt.show()


    # Plots Reconstruction Error (y) over Compression (x), at different Bases (colorbar) as a bar graph.
def plotbar_cs_basis(y1, y2, y3, y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='fig.pdf', bbox=(0,0,1,1), ylim=None, cspercentage=False):
    fig, ax = plt.subplots(figsize=sizefig)
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
        y4 = (y4 - y4.min()) / (y4.max()-y4.min())
    
    # Express cs_ratios as decimals instead
    if(cspercentage==False):
        cs_ratios_ = ("0.25", "0.50", "0.75")
        basis_dict = {
            'DCT': y1,
            'Haar DWT': y2,
            'Db-2 DWT': y3,
            'Coif-1 DWT': y4,
            
        }

        x = np.arange(len(cs_ratios_))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0
        
        #fig, ax = plt.subplots(layout='constrained')
        fig, ax = plt.subplots(figsize=sizefig)
        
        for attribute, measurement in basis_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset-0.0955, measurement, width, label=attribute)
            #ax.bar_label(rects, padding=3, size=10)
            multiplier += 1
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
        ax.tick_params(axis='both',labelsize=labelsize+4)
        ax.set_xticks(x + width, cs_ratios_)
        ax.set_xlabel('Compression Ratio', fontsize=labelsize+4, labelpad=6)
        ax.legend(loc=loclegend, title='Basis', fontsize=labelsize+3, title_fontproperties={'size':labelsize+4}, bbox_to_anchor=bbox, ncol=2)
        ax.set_ylim(ylim)
    # Express cs_ratios as percentage instead
    else:
        cs_ratios_ = ("25%", "50%", "75%")
        basis_dict = {
            'DCT': y1,
            'Haar DWT': y2,
            'Db-2 DWT': y3,
            'Coif-1 DWT': y4,
            
        }
    
        x = np.arange(len(cs_ratios_))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0
        
        #fig, ax = plt.subplots(layout='constrained')
        fig, ax = plt.subplots(figsize=sizefig)
        
        for attribute, measurement in basis_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset-0.0955, measurement, width, label=attribute)
            #ax.bar_label(rects, padding=3, size=10)
            multiplier += 1
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
        ax.tick_params(axis='both',labelsize=labelsize+4)
        ax.set_xticks(x + width, cs_ratios_)
        ax.set_xlabel('Compression Ratio [%]', fontsize=labelsize+4, labelpad=6)
        ax.legend(loc=loclegend, title='Basis', fontsize=labelsize+3, title_fontproperties={'size':labelsize+4}, bbox_to_anchor=bbox, ncol=2)
        ax.set_ylim(ylim)
    '''
    if(title != ''):
        ax.set_title(title, size=labelsize)
    '''
    plt.savefig(os.path.join(OUTPUT_PATH_PLOTS, figname), format='pdf', bbox_inches='tight')
    plt.show()
    
''' ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' Same plotting functions but adjusted to include 10% Compression Rate too'''

def plot_cs_basis_10csratio(y1, y2, y3, y4, normalized=True, xlabel='testx', ylabel='testy', title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='fig.pdf', bbox=(0,0,1,1), ylim=None):
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
        y4 = (y4 - y4.min()) / (y4.max()-y4.min())
    
    fig, ax = plt.subplots(figsize=sizefig)
    
    #ax.scatter(x,y, marker='o')
    ax.plot(cs_ratios, y1, linestyle='-', marker='o', label='DCT', linewidth=3, markeredgewidth=2)
    ax.plot(cs_ratios, y2, linestyle='-', marker='o', label='Haar DWT', linewidth=3, markeredgewidth=2)
    ax.plot(cs_ratios, y3, linestyle='-', marker='o', label='Db2 DWT', linewidth=3, markeredgewidth=2)
    ax.plot(cs_ratios, y4, linestyle='-', marker='o', label='Coif1 DWT', linewidth=3, markeredgewidth=2)
    ax.legend(loc=loclegend, title='Basis', fontsize=labelsize, title_fontsize=labelsize+1, bbox_to_anchor=bbox)
    #ax2 = ax.twinx()
    #plt.rcParams['axes.labelsize'] = 20
    ax.tick_params(axis='both',labelsize=labelsize)
    ax.set_xlabel('Compression [%]', fontsize=labelsize)
    ax.set_ylabel('MAE', fontsize=labelsize)
    ax.set_ylim(ylim)
    #if(title != ''):
    #    ax.set_title(title, size=labelsize)
    
    ax.set_xticks([10, 25, 50, 75])
    plt.savefig(os.path.join(OUTPUT_PATH_PLOTS, figname), format='pdf', bbox_inches='tight')
    plt.show()


    # Same basis, different sparsity, different cs ratios
def plot_cs_sparsity_10csratio(y1, y2, y3, y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='fig.pdf', bbox=(0,0,1,1), ylim=None):
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
        y4 = (y4 - y4.min()) / (y4.max()-y4.min())
    
    fig, ax = plt.subplots(figsize=sizefig)
    
    #ax.scatter(x,y, marker='o')
    ax.plot(sparsities, y1, linestyle='-', marker='o', label='10%', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y2, linestyle='-', marker='o', label='25%', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y3, linestyle='-', marker='o', label='50%', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y4, linestyle='-', marker='o', label='75%', linewidth=3, markeredgewidth=2)
    ax.legend(loc=loclegend, title='Compression Ratio', fontsize=labelsize+3, title_fontsize=labelsize+4, bbox_to_anchor=bbox)  
    #ax2 = ax.twinx() 
    #plt.rcParams['axes.labelsize'] = 20
    ax.tick_params(axis='both',labelsize=labelsize+4)
    ax.set_xlabel('Sparsity [%]', fontsize=labelsize+4, labelpad=6)
    ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
    ax.set_ylim(ylim)
    '''
    if(title != ''):
        ax.set_title(title, size=labelsize)
    '''
    plt.savefig(os.path.join(OUTPUT_PATH_PLOTS, figname), format='pdf', bbox_inches='tight')
    plt.show()


    # Same cs ratio, different sparsity, different basis
def plot_basis_sparsity_10csratio(y1, y2, y3, y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='fig.pdf', bbox=(0,0,1,1), ylim=None):
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
        y4 = (y4 - y4.min()) / (y4.max()-y4.min())
    
    fig, ax = plt.subplots(figsize=sizefig)
    
    # Choose for which CS Ratio to plot here!
    
    #ax.scatter(x,y, marker='o')
    ax.plot(sparsities, y1, linestyle='-', marker='o', label='DCT', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y2, linestyle='-', marker='o', label='Haar DWT', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y3, linestyle='-', marker='o', label='Db2 DWT', linewidth=3, markeredgewidth=2)
    ax.plot(sparsities, y4, linestyle='-', marker='o', label='Coif1 DWT', linewidth=3, markeredgewidth=2)
    ax.legend(loc=loclegend, title='Basis', fontsize=labelsize+4, title_fontsize=labelsize+4, bbox_to_anchor=bbox)  
    #ax2 = ax.twinx()
    #plt.rcParams['axes.labelsize'] = 20
    ax.tick_params(axis='both',labelsize=labelsize+4)
    ax.set_xlabel('Sparsity [%]', fontsize=labelsize+3, labelpad=6)
    ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
    ax.set_ylim(ylim)
    '''
    if(title != ''):
        ax.set_title(title, size=labelsize)
    '''
    plt.savefig(os.path.join(OUTPUT_PATH_PLOTS, figname), format='pdf', bbox_inches='tight')
    plt.show()


def plotbar_cs_basis_10csratio(y1,y2,y3,y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='fig.pdf', bbox=(0,0,1,1), ylim=None, cspercentage=False):
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
        y4 = (y4 - y4.min()) / (y4.max()-y4.min())
    
    fig, ax = plt.subplots(figsize=sizefig)
    
    if(cspercentage==False):
        cs_ratios_ = ("0.10", "0.25", "0.50", "0.75")
        basis_dict = {
            'DCT': y1,
            'Haar DWT': y2,
            'Db-2 DWT': y3,
            'Coif-1 DWT': y4,
            
        }

        x = np.arange(len(cs_ratios_))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0
        
        #fig, ax = plt.subplots(layout='constrained')
        fig, ax = plt.subplots(figsize=sizefig)
        
        for attribute, measurement in basis_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset-0.0955, measurement, width, label=attribute)
            #ax.bar_label(rects, padding=3, size=10)
            multiplier += 1
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
        ax.tick_params(axis='both',labelsize=labelsize+4)
        ax.set_xticks(x + width, cs_ratios_)
        ax.set_xlabel('Compression Ratio', fontsize=labelsize+4, labelpad=6)
        ax.legend(loc=loclegend, title='Basis', fontsize=labelsize+3, title_fontproperties={'size':labelsize+4}, bbox_to_anchor=bbox, ncol=2)
        ax.set_ylim(ylim)
    else:
        cs_ratios_ = ("10%", "25%", "50%", "75%")
        basis_dict = {
            'DCT': y1,
            'Haar DWT': y2,
            'Db-2 DWT': y3,
            'Coif-1 DWT': y4,
            
        }
    
        x = np.arange(len(cs_ratios_))  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0
        
        #fig, ax = plt.subplots(layout='constrained')
        fig, ax = plt.subplots(figsize=sizefig)
        
        for attribute, measurement in basis_dict.items():
            offset = width * multiplier
            rects = ax.bar(x + offset-0.0955, measurement, width, label=attribute)
            #ax.bar_label(rects, padding=3, size=10)
            multiplier += 1
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('MAE', fontsize=labelsize+4, labelpad=8)
        ax.tick_params(axis='both',labelsize=labelsize+4)
        ax.set_xticks(x + width, cs_ratios_)
        ax.set_xlabel('Compression [%]', fontsize=labelsize+4, labelpad=6)
        ax.legend(loc=loclegend, title='Basis', fontsize=labelsize+3, title_fontproperties={'size':labelsize+4}, bbox_to_anchor=bbox, ncol=2)
        ax.set_ylim(ylim)
    '''
    if(title != ''):
        ax.set_title(title, size=labelsize)
    '''
    plt.savefig(os.path.join(OUTPUT_PATH_PLOTS, figname), format='pdf', bbox_inches='tight')
    plt.show()











''' --------------------------------------------------------------- '''



if __name__ == "__main__":
    
    
    # Choose the basis to use (DWT, coif1?)
    # Plots Reconstruction Error (y) over Sparsity (x), at different Compression Ratios (colorbar) as a line graph.
    plot_cs_sparsity(MtMarcyMAE_cs25, MtMarcyMAE_cs50, MtMarcyMAE_cs75, title='Reconstruction With Daubechies-2 DWT of Mt. Marcy', bbox=(1,1), figname='MtMarcyDb2CS_v7_ratio.pdf') # bbox=(0,0.925-0.0165)
    plot_cs_sparsity(DeerparkMAE_cs25, DeerparkMAE_cs50, DeerparkMAE_cs75, title='Reconstruction With Daubechies-2 DWT of Deerpark', bbox=(1,1), figname='DeerparkDb2CS_v7_ratio.pdf') # bbox=(0,0.7)
    plot_cs_sparsity(ColCircMAE_cs25, ColCircMAE_cs50, ColCircMAE_cs75, title='Reconstruction Using DCT of Columbus Circle', bbox=(1,1), figname='ColCircDCTCS_v7_ratio.pdf') # bbox=(0+0.35,0.95+0.002)
    plot_cs_sparsity(ParkslopeMAE_cs25, ParkslopeMAE_cs50, ParkslopeMAE_cs75, title='Reconstruction Using DCT of Park Slope', bbox=(1,1), figname='ParkslopeDCTCS_v7_ratio.pdf') # bbox=(0,0.9)
    
    # Choose the CS Ratio to use (25%)
    # Plots Reconstruction Error (y) over Sparsity (x), at different Bases (colorbar) as a line graph.
    plot_basis_sparsity(MtMarcyMAE_dct, MtMarcyMAE_haar, MtMarcyMAE_db2, MtMarcyMAE_coif1, title='Reconstruction With 25% Compression of Mt. Marcy', bbox=(0,0.90+0.02), figname='MtMarcy25CSBases_v6.pdf')
    plot_basis_sparsity(DeerparkMAE_dct, DeerparkMAE_haar, DeerparkMAE_db2, DeerparkMAE_coif1, title='Reconstruction With 25% Compression of Deerpark', bbox=(0.65,1), figname='Deerpark25CSBases_v6.pdf')
    plot_basis_sparsity(ColCircMAE_dct, ColCircMAE_haar, ColCircMAE_db2, ColCircMAE_coif1, title='Reconstruction With 25% Compression of Columbus Circle', loclegend='lower left', figname='ColCirc25CSBases_v6.pdf')
    plot_basis_sparsity(ParkslopeMAE_dct, ParkslopeMAE_haar, ParkslopeMAE_db2, ParkslopeMAE_coif1, title='Reconstruction With 25% Compression of Park Slope', loclegend='lower left', figname='Parkslope25CSBases_v6.pdf')
    
    # Choose the sparsity to use (80?, 90% instead now)
    # Plots Reconstruction Error (y) over Compression (x), at different Bases (colorbar) as a line graph.
    plot_cs_basis(MtMarcyMAE_dct_s, MtMarcyMAE_haar_s, MtMarcyMAE_db2_s, MtMarcyMAE_coif1_s, title='Reconstruction With 90% Sparsity of Mt. Marcy', loclegend='upper right', figname='MtMarcy90SparsityBases_v6_ratio.pdf')
    plot_cs_basis(DeerparkMAE_dct_s, DeerparkMAE_haar_s, DeerparkMAE_db2_s, DeerparkMAE_coif1_s, title='Reconstruction With 90% Sparsity of Deerpark', loclegend='upper right', figname='Deerpark90SparsityBases_v6_ratio.pdf')
    plot_cs_basis(ColCircMAE_dct_s, ColCircMAE_haar_s, ColCircMAE_db2_s, ColCircMAE_coif1_s, title='Reconstruction with 90% Sparsity of Columbus Circle', loclegend='upper right', figname='ColCirc90SparsityBases_v6_ratio.pdf')
    plot_cs_basis(ParkslopeMAE_dct_s, ParkslopeMAE_haar_s, ParkslopeMAE_db2_s, ParkslopeMAE_coif1_s, title='Reconstruction with 90% Sparsity of Park Slope', loclegend='upper right', figname='Parkslope90SparsityBases_v6_ratio.pdf')
    
    # Bar Graph
    # Plots Reconstruction Error (y) over Compression (x), at different Bases (colorbar) as a bar graph.
    plotbar_cs_basis(MtMarcyMAE_dct_s, MtMarcyMAE_haar_s, MtMarcyMAE_db2_s, MtMarcyMAE_coif1_s, title='Reconstruction With 90% Sparsity of Mt. Marcy', loclegend='upper right', figname='MtMarcy90SparsityBases_Bar_v7_ratio.pdf')
    plotbar_cs_basis(DeerparkMAE_dct_s, DeerparkMAE_haar_s, DeerparkMAE_db2_s, DeerparkMAE_coif1_s, title='Reconstruction With 90% Sparsity of Deerpark', loclegend='upper right', figname='Deerpark90SparsityBases_Bar_v7_ratio.pdf')
    plotbar_cs_basis(ColCircMAE_dct_s, ColCircMAE_haar_s, ColCircMAE_db2_s, ColCircMAE_coif1_s, title='Reconstruction With 90% Sparsity of Columbus Circle', loclegend='upper right', figname='ColCirc90SparsityBases_Bar_v7_ratio.pdf')
    plotbar_cs_basis(ParkslopeMAE_dct_s, ParkslopeMAE_haar_s, ParkslopeMAE_db2_s, ParkslopeMAE_coif1_s, title='Reconstruction With 90% Sparsity of Park Slope', loclegend='upper right', figname='Parkslope90SparsityBases_Bar_v7_ratio.pdf')
    
    
    
    ''' ----------------------------------- '''
    # ADDING 10% CS RATIO: --------------------------------------------------------------------------------------
    '''
    plot_cs_sparsity_10csratio(MtMarcyMAE_cs10, MtMarcyMAE_cs25, MtMarcyMAE_cs50, MtMarcyMAE_cs75, title='Reconstruction With Daubechies-2 DWT of Mt. Marcy', bbox=(1,1), figname='MtMarcyDb2CS_updated_v1_ratio.pdf', ylim=(0, 85))
    plot_cs_sparsity_10csratio(DeerparkMAE_cs10, DeerparkMAE_cs25, DeerparkMAE_cs50, DeerparkMAE_cs75, title='Reconstruction With Daubechies-2 DWT of Deerpark', bbox=(1,1), figname='DeerparkDb2CS_updated_v1_ratio.pdf', ylim=(0, 130))
    plot_cs_sparsity_10csratio(ColCircMAE_cs10, ColCircMAE_cs25, ColCircMAE_cs50, ColCircMAE_cs75, title='Reconstruction Using DCT of Columbus Circle', bbox=(1,1), figname='ColCircDCTCS_updated_v1_ratio.pdf', ylim=(0, 95))
    plot_cs_sparsity_10csratio(ParkslopeMAE_cs10, ParkslopeMAE_cs25, ParkslopeMAE_cs50, ParkslopeMAE_cs75, title='Reconstruction Using DCT of Park Slope', bbox=(1,1), figname='ParkslopeDCTCS_updated_v1_ratio.pdf', ylim=(0, 45))
    
    plot_cs_basis_10csratio(MtMarcyMAE_dct_s, MtMarcyMAE_haar_s, MtMarcyMAE_db2_s, MtMarcyMAE_coif1_s, title='Reconstruction With 90% Sparsity of Mt. Marcy', loclegend='upper right', figname='MtMarcy90SparsityBases_updated_v1_ratio.pdf')
    plot_cs_basis_10csratio(DeerparkMAE_dct_s, DeerparkMAE_haar_s, DeerparkMAE_db2_s, DeerparkMAE_coif1_s, title='Reconstruction With 90% Sparsity of Deerpark', loclegend='upper right', figname='Deerpark90SparsityBases_updated_v1_ratio.pdf')
    plot_cs_basis_10csratio(ColCircMAE_dct_s, ColCircMAE_haar_s, ColCircMAE_db2_s, ColCircMAE_coif1_s, title='Reconstruction with 90% Sparsity of Columbus Circle', loclegend='upper right', figname='ColCirc90SparsityBases_updated_v1_ratio.pdf')
    plot_cs_basis_10csratio(ParkslopeMAE_dct_s, ParkslopeMAE_haar_s, ParkslopeMAE_db2_s, ParkslopeMAE_coif1_s, title='Reconstruction with 90% Sparsity of Park Slope', loclegend='upper right', figname='Parkslope90SparsityBases_updated_v1_ratio.pdf')
    
    plot_basis_sparsity_10csratio(MtMarcyMAE_dct, MtMarcyMAE_haar, MtMarcyMAE_db2, MtMarcyMAE_coif1, title='Reconstruction With 10% Compression of Mt. Marcy', bbox=(0,0.90+0.02), figname='MtMarcy10CSBases_updated_v1.pdf')
    plot_basis_sparsity_10csratio(DeerparkMAE_dct, DeerparkMAE_haar, DeerparkMAE_db2, DeerparkMAE_coif1, title='Reconstruction With 10% Compression of Deerpark', bbox=(0.65,1), figname='Deerpark10CSBases_updated_v1.pdf')
    plot_basis_sparsity_10csratio(ColCircMAE_dct, ColCircMAE_haar, ColCircMAE_db2, ColCircMAE_coif1, title='Reconstruction With 10% Compression of Columbus Circle', loclegend='lower left', figname='ColCirc10CSBases_updated_v1.pdf')
    plot_basis_sparsity_10csratio(ParkslopeMAE_dct, ParkslopeMAE_haar, ParkslopeMAE_db2, ParkslopeMAE_coif1, title='Reconstruction With 10% Compression of Park Slope', loclegend='lower left', figname='Parkslope10CSBases_updated_v1.pdf')
   
    plotbar_cs_basis_10csratio(MtMarcyMAE_dct_s, MtMarcyMAE_haar_s, MtMarcyMAE_db2_s, MtMarcyMAE_coif1_s, title='Reconstruction With 90% Sparsity of Mt. Marcy', loclegend='upper right', figname='MtMarcy90SparsityBases_Bar_updated_v1_ratio.pdf')
    plotbar_cs_basis_10csratio(DeerparkMAE_dct_s, DeerparkMAE_haar_s, DeerparkMAE_db2_s, DeerparkMAE_coif1_s, title='Reconstruction With 90% Sparsity of Deerpark', loclegend='upper right', figname='Deerpark90SparsityBases_Bar_updated_v1_ratio.pdf')
    plotbar_cs_basis_10csratio(ColCircMAE_dct_s, ColCircMAE_haar_s, ColCircMAE_db2_s, ColCircMAE_coif1_s, title='Reconstruction With 90% Sparsity of Columbus Circle', loclegend='upper right',  ylim=(0,100), figname='ColCirc90SparsityBases_Bar_updated_v1_ratio.pdf')
    plotbar_cs_basis_10csratio(ParkslopeMAE_dct_s, ParkslopeMAE_haar_s, ParkslopeMAE_db2_s, ParkslopeMAE_coif1_s, title='Reconstruction With 90% Sparsity of Park Slope', loclegend='upper right',  ylim=(0,50), figname='Parkslope90SparsityBases_Bar_updated_v1_ratio.pdf')
    '''