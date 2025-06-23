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

# Notes: Coif1 seems to work slightly better for Mt Marcy
# But Db1 seems to work slightly better for Columbus Circle






# w.r.t sparsity: (cs_ratio constant)
# Choose which cs_ratio to use! (I chose 25%)

sparsities = np.array(          [10,      20,      30,      40,      50,      60,      70,      80,      90])
MtMarcyMAE_dct = np.array(      [52.8763, 52.1364, 52.1330, 51.4397, 52.4747, 51.2110, 51.2954, 48.3100, 41.2405])
MtMarcyMAE_haar = np.array(     [37.9475, 38.4487, 39.2984, 38.0599, 38.4315, 37.7491, 36.0268, 36.8038, 33.8705])
MtMarcyMAE_db2 = np.array(      [35.0337, 33.8074, 34.7388, 34.6996, 35.0455, 35.6748, 34.0143, 32.4462, 29.3805])
MtMarcyMAE_coif1 = np.array(    [34.9028, 36.4107, 34.8897, 35.3826, 34.0483, 35.4986, 33.9462, 33.3784, 29.7862])

DeerparkMAE_dct = np.array(     [69.2727, 66.0980, 65.8345, 67.0540, 67.9450, 68.4320, 66.3139, 62.3310, 57.9150])
DeerparkMAE_haar = np.array(    [64.0353, 66.7063, 61.8269, 59.2558, 64.4545, 63.7639, 65.6325, 60.6544, 62.4718])
DeerparkMAE_db2 = np.array(     [61.4427, 57.2948, 57.9210, 64.4112, 57.2199, 56.5267, 61.1169, 66.4953, 55.4912])
DeerparkMAE_coif1 = np.array(   [57.4789, 54.3365, 53.6098, 61.2682, 61.3423, 54.0504, 58.7286, 62.0088, 58.6857])

ColCircMAE_dct = np.array(      [70.8113, 72.1764, 71.6407, 73.0444, 72.1836, 72.3669, 70.6031, 67.3215, 62.6746])
ColCircMAE_haar = np.array(     [74.6002, 74.7281, 74.4680, 73.3981, 74.1170, 74.2936, 71.7552, 69.8373, 65.4650])
ColCircMAE_db2 = np.array(      [71.7215, 73.5260, 73.3655, 72.9085, 71.7870, 72.9229, 71.8965, 70.0874, 63.9326])
ColCircMAE_coif1 = np.array(    [72.3120, 74.0243, 71.1795, 71.9102, 73.2416, 72.2631, 71.8514, 69.5189, 64.9144])

ParkslopeMAE_dct = np.array(      [31.4273, 31.3195, 31.6604, 31.9479, 30.6764, 30.8481, 29.7346, 29.6771, 28.2300])
ParkslopeMAE_haar = np.array(     [36.3097, 36.2351, 35.6291, 36.2691, 35.2802, 35.1863, 34.8515, 34.1669, 31.7954])
ParkslopeMAE_db2 = np.array(      [34.6049, 34.1019, 34.8003, 34.3365, 34.1815, 33.5670, 33.6506, 32.6905, 30.4399])
ParkslopeMAE_coif1 = np.array(    [33.8523, 34.1886, 34.3686, 34.1893, 34.5081, 34.0832, 33.6576, 32.6409, 30.7774])

# w.r.t sparsity: (cs_ratio constant)
# Choose which cs_ratio to use! (I chose 10%)
'''
sparsities = np.array(          [10,      20,      30,      40,      50,      60,      70,      80,      90])
MtMarcyMAE_dct = np.array(      [92.3353, 82.2806, 90.7726, 83.6298, 86.7067, 89.0867, 89.7059, 80.9692, 86.1067])
MtMarcyMAE_haar = np.array(     [88.2085, 85.0063, 78.0144, 79.7248, 75.5201, 80.6679, 81.6293, 81.8747, 73.8393])
MtMarcyMAE_db2 = np.array(      [67.4344, 72.2146, 79.5638, 69.4901, 70.9503, 73.7126, 70.9036, 78.1377, 66.5664])
MtMarcyMAE_coif1 = np.array(    [72.4171, 76.4990, 80.5334, 80.4529, 78.4834, 78.3172, 74.4746, 71.7682, 68.2379])

DeerparkMAE_dct = np.array(     [194.0459, 189.2413, 197.9364, 201.3991, 205.2373, 194.3512, 196.5156, 190.3431, 201.874])
DeerparkMAE_haar = np.array(    [233.9342, 234.7283, 231.5110, 233.6409, 233.6126, 228.5895, 236.8978, 237.4573, 238.3772])
DeerparkMAE_db2 = np.array(     [253.8411, 237.9315, 258.5657, 246.1605, 251.5261, 252.1233, 245.2826, 254.0323, 247.3628])
DeerparkMAE_coif1 = np.array(   [269.2157, 250.9539, 257.5588, 247.8834, 262.6258, 251.6119, 263.0036, 272.9754, 258.0684])

ColCircMAE_dct = np.array(      [90.0385, 87.2606, 87.2060, 87.3391, 87.8879, 86.0571, 87.9613, 83.2628, 80.2309])
ColCircMAE_haar = np.array(     [87.5147, 86.0479, 87.5292, 87.4626, 85.2768, 86.2611, 87.2931, 83.7641, 79.7570])
ColCircMAE_db2 = np.array(      [87.8417, 87.7746, 90.1551, 90.6316, 87.1895, 87.5790, 86.6486, 84.0952, 83.5343])
ColCircMAE_coif1 = np.array(    [86.8454, 90.6986, 90.7152, 89.5150, 88.2282, 90.1954, 91.0456, 87.9761, 83.0095])

ParkslopeMAE_dct = np.array(      [43.8712, 42.5156, 42.0670, 42.8551, 41.8692, 42.7081, 43.6765, 43.2374, 41.6998])
ParkslopeMAE_haar = np.array(     [45.1320, 45.4045, 44.6514, 44.9215, 44.6118, 43.0186, 43.0880, 42.2622, 41.3472])
ParkslopeMAE_db2 = np.array(      [41.2486, 40.9739, 41.4346, 40.4508, 40.7150, 41.2638, 40.0648, 39.9186, 37.7577])
ParkslopeMAE_coif1 = np.array(    [41.3511, 40.4843, 41.1081, 43.2589, 42.4775, 41.8934, 42.3498, 39.4401, 38.9559])
'''







# w.r.t cs_ratios: (sparsity constant)
# Choose which sparsity to use! (I chose 80% for Mt Marcy and 80% for Columbus Circle)
'''
cs_ratios_decimal = np.array(     [0.25,      0.50,      0.75])
cs_ratios = np.array(           [25,      50,      75])
MtMarcyMAE_dct_s = np.array(      [48.31, 21.5537, 16.1825])
MtMarcyMAE_haar_s = np.array(     [36.8038, 16.3898, 11.6712])
MtMarcyMAE_db2_s = np.array(      [32.4462, 15.3936, 11.5696])
MtMarcyMAE_coif1_s = np.array(    [33.3784, 16.1105, 11.7246])
DeerparkMAE_dct_s = np.array(      [])
DeerparkMAE_haar_s = np.array(     [])
DeerparkMAE_db2_s = np.array(      [])
DeerparkMAE_coif1_s = np.array(    [])
ColCircMAE_dct_s = np.array(      [67.3215, 40.5298, 31.5694])
ColCircMAE_haar_s = np.array(     [69.8373, 40.9653, 31.9231])
ColCircMAE_db2_s = np.array(      [70.0874, 39.5118, 28.637])
ColCircMAE_coif1_s = np.array(    [69.5189, 39.7269, 28.6126])
ParkslopeMAE_dct_s = np.array(      [])
ParkslopeMAE_haar_s = np.array(     [])
ParkslopeMAE_db2_s = np.array(      [])
ParkslopeMAE_coif1_s = np.array(    [])
'''

# w.r.t cs_ratios: (sparsity constant)
# Choose which sparsity to use! (I chose 90% for Mt Marcy and 90% for Columbus Circle)

cs_ratios_decimal = np.array(   [0.25,   0.50,     0.75])
cs_ratios = np.array(             [25,      50,      75])
MtMarcyMAE_dct_s = np.array(      [41.2405, 21.6617, 21.6618])
MtMarcyMAE_haar_s = np.array(     [33.8705, 18.0075, 18.0074])
MtMarcyMAE_db2_s = np.array(      [29.3805, 17.1784, 17.1784])
MtMarcyMAE_coif1_s = np.array(    [29.7862, 17.4388, 17.4388])
DeerparkMAE_dct_s = np.array(     [57.9150, 27.1976, 27.1975])
DeerparkMAE_haar_s = np.array(    [62.4718, 19.0685, 19.0685])
DeerparkMAE_db2_s = np.array(     [55.4912, 11.2643, 11.2643])
DeerparkMAE_coif1_s = np.array(   [58.6857, 10.1783, 10.1783])
ColCircMAE_dct_s = np.array(      [62.6746, 42.2140, 42.2140])
ColCircMAE_haar_s = np.array(     [65.4650, 45.8213, 45.8213])
ColCircMAE_db2_s = np.array(      [70.0874, 42.5750, 42.5750])
ColCircMAE_coif1_s = np.array(    [64.9144, 41.9929, 41.9929])
ParkslopeMAE_dct_s = np.array(    [28.2300, 13.4230, 13.4230])
ParkslopeMAE_haar_s = np.array(   [31.7954, 25.2846, 25.2846])
ParkslopeMAE_db2_s = np.array(    [30.4399, 22.3706, 22.3706])
ParkslopeMAE_coif1_s = np.array(  [30.7774, 21.7863, 21.7863])


# w.r.t cs_ratios: (sparsity constant)
# Choose which sparsity to use! (I chose 90% for Mt Marcy and 90% for Columbus Circle)
'''
cs_ratios_decimal = np.array(   [0.10,    0.25,   0.50,     0.75])
cs_ratios = np.array(             [10,      25,      50,      75])
MtMarcyMAE_dct_s = np.array(      [86.1067, 41.2405, 21.6617, 21.6618])
MtMarcyMAE_haar_s = np.array(     [73.8393, 33.8705, 18.0075, 18.0074])
MtMarcyMAE_db2_s = np.array(      [66.5664, 29.3805, 17.1784, 17.1784])
MtMarcyMAE_coif1_s = np.array(    [68.2379, 29.7862, 17.4388, 17.4388])
DeerparkMAE_dct_s = np.array(     [99.1596, 57.9150, 27.1976, 27.1975])
DeerparkMAE_haar_s = np.array(    [119.6452,62.4718, 19.0685, 19.0685])
DeerparkMAE_db2_s = np.array(     [119.9499,55.4912, 11.2643, 11.2643])
DeerparkMAE_coif1_s = np.array(   [124.5423,58.6857, 10.1783, 10.1783])
ColCircMAE_dct_s = np.array(      [80.2309,62.6746, 42.2140, 42.2140])
ColCircMAE_haar_s = np.array(     [79.757,65.4650, 45.8213, 45.8213])
ColCircMAE_db2_s = np.array(      [83.5343,70.0874, 42.5750, 42.5750])
ColCircMAE_coif1_s = np.array(    [83.0095,64.9144, 41.9929, 41.9929])
ParkslopeMAE_dct_s = np.array(    [41.6998,28.2300, 13.4230, 13.4230])
ParkslopeMAE_haar_s = np.array(   [41.3472,31.7954, 25.2846, 25.2846])
ParkslopeMAE_db2_s = np.array(    [37.7577,30.4399, 22.3706, 22.3706])
ParkslopeMAE_coif1_s = np.array(  [38.9559,30.7774, 21.7863, 21.7863])
'''






# w.r.t. sparsity and cs_ratios (basis constant)
# Choose which basis to use! (I choose Coiflets-1)
'''
sparsities = np.array(      [40,      50,      60,       70,      80,      90])
MtMarcyMAE_cs25 = np.array( [35.3826, 34.0483, 35.4986, 33.9462, 33.3784, 29.7862])
MtMarcyMAE_cs50 = np.array( [21.5306, 21.2314, 20.7696, 19.7802, 16.1105, 17.4388])
MtMarcyMAE_cs75 = np.array( [11.35, 10.5518, 6.6538, 8.083, 11.7246, 17.4388])
DeerparkMAE_cs25 = np.array(      [])
DeerparkMAE_cs50 = np.array(     [])
DeerparkMAE_cs75 = np.array(      [])
ColCircMAE_cs25 = np.array( [71.9102, 73.2416, 72.2631, 71.8514, 69.5189, 64.9144])
ColCircMAE_cs50 = np.array( [52.2703, 52.3205, 49.8137, 48.1756, 39.7269, 41.9929])
ColCircMAE_cs75 = np.array( [27.4362, 27.4333, 13.2792, 19.7161, 28.6126, 41.9929])
ParkslopeMAE_cs25 = np.array(      [])
ParkSlopeMAE_cs50 = np.array(     [])
ParkSlopeMAE_cs75 = np.array(      [])
'''

# w.r.t. sparsity and cs_ratios (basis constant)
# Choose which basis to use! (I choose Db2 for Mt Marcy and Deerpark and DCT for Columbus Circle and Parkslope)
# Db2
sparsities = np.array(      [10,      20,      30,      40,      50,      60,      70,      80,      90])
MtMarcyMAE_cs10 = np.array( [67.4344, 72.2146, 79.5638, 69.4901, 70.9503, 73.7126, 70.9036, 78.1377, 66.5664])
MtMarcyMAE_cs25 = np.array( [35.0337, 33.8074, 34.7388, 34.6996, 35.0455, 35.6748, 34.0143, 32.4462, 29.3805])
MtMarcyMAE_cs50 = np.array( [21.2567, 21.9177, 21.5277, 21.3140, 20.9998, 20.4355, 19.7295, 15.3936, 17.1784])
MtMarcyMAE_cs75 = np.array( [11.4037, 11.6332, 11.6731, 11.2760, 10.5023,  5.7799,  8.0703, 11.5696, 17.1784])
DeerparkMAE_cs10 = np.array([124.2548, 116.6501, 129.7318, 118.6416, 122.4280, 121.8614, 120.4470, 125.3111, 119.9499])
DeerparkMAE_cs25 = np.array([61.4427, 57.2948, 57.9210, 64.4112, 57.2199, 56.5267, 61.1169, 66.4953, 55.4912])
DeerparkMAE_cs50 = np.array([10.2686, 10.3698, 9.8943, 10.5782, 10.1792, 9.6382, 9.0404, 7.3963, 11.2643])
DeerparkMAE_cs75 = np.array([5.8718,  5.7936,  5.7492, 5.5305,  5.2699,  3.6999, 4.2868, 5.6229, 11.2642])
# DCT
ColCircMAE_cs10 = np.array(  [90.0385, 87.2606, 87.2060, 87.3391, 87.8879, 86.0571, 87.9613, 83.2628, 80.2309])
ColCircMAE_cs25 = np.array(  [70.8113, 72.1764, 71.6407, 73.0444, 72.1836, 72.3669, 70.6031, 67.3215, 62.6746])
ColCircMAE_cs50 = np.array(  [52.8574, 53.3964, 53.0441, 51.8429, 50.8054, 50.2189, 48.5305, 40.5298, 42.2140])
ColCircMAE_cs75 = np.array(  [34.5028, 33.4325, 33.7122, 32.2317, 29.8566, 21.6500, 24.1223, 31.5694, 42.2140])
ParkslopeMAE_cs10 = np.array([43.8712, 42.5156, 42.0670, 42.8551, 41.8692, 42.7081, 43.6765, 43.2374, 41.6998])
ParkslopeMAE_cs25 = np.array([31.4273, 31.3195, 31.6604, 31.9479, 30.6764, 30.6764, 29.7346, 29.6771, 28.2300])
ParkslopeMAE_cs50 = np.array([17.7062, 17.0823, 17.3362, 17.6088, 17.2914, 16.2458, 15.4149, 12.1865, 13.4230])
ParkslopeMAE_cs75 = np.array([9.7533,  9.9737,   9.7332,  9.6748,  8.6984,  6.4142,  6.6052,  8.5875, 13.4230])







    # Same sparsity, different cs ratios, different basis
def plot_cs_basis(y1, y2, y3, y4, normalized=True, xlabel='testx', ylabel='testy', title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='D:/Downloads/fig.pdf', bbox=(0,0,1,1), ylim=None, cspercentage=False):
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
        ax.set_xlabel('Compression', fontsize=labelsize)
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
        ax.set_xlabel('Compression [%]', fontsize=labelsize)
        ax.set_ylabel('MAE', fontsize=labelsize)
        ax.set_ylim(ylim)
        if(title != ''):
            ax.set_title(title, size=labelsize)
        ax.set_xticks([25, 50, 75])
        
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    plt.show()
    
    # Same cs ratio, different sparsity, different basis
def plot_basis_sparsity(y1, y2, y3, y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='D:/Downloads/fig.pdf', bbox=(0,0,1,1), ylim=None):
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
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    plt.show()
    
    # Same basis, different sparsity, different cs ratios
def plot_cs_sparsity(y1, y2, y3, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='D:/Downloads/fig.pdf', bbox=(0,0,1,1), ylim=None, cspercentage=False):
    
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
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    plt.show()


def plotbar_cs_basis(y1, y2, y3, y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='D:/Downloads/fig.pdf', bbox=(0,0,1,1), ylim=None, cspercentage=False):
    fig, ax = plt.subplots(figsize=sizefig)
    
    # Min-max normalizing data between (0,1) range.
    if(normalized):
        y1 = (y1 - y1.min()) / (y1.max()-y1.min())
        y2 = (y2 - y2.min()) / (y2.max()-y2.min())
        y3 = (y3 - y3.min()) / (y3.max()-y3.min())
        y4 = (y4 - y4.min()) / (y4.max()-y4.min())
    
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
        ax.set_xlabel('Compression [%]', fontsize=labelsize+4, labelpad=6)
        ax.legend(loc=loclegend, title='Basis', fontsize=labelsize+3, title_fontproperties={'size':labelsize+4}, bbox_to_anchor=bbox, ncol=2)
        ax.set_ylim(ylim)
    '''
    if(title != ''):
        ax.set_title(title, size=labelsize)
    '''
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    plt.show()
    
''' ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- '''
''' Same plotting functions but adjusted to include 10% Compression Rate too'''

def plot_cs_basis_10csratio(y1, y2, y3, y4, normalized=True, xlabel='testx', ylabel='testy', title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='D:/Downloads/fig.pdf', bbox=(0,0,1,1), ylim=None):
    
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
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    plt.show()


    # Same basis, different sparsity, different cs ratios
def plot_cs_sparsity_10csratio(y1, y2, y3, y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='D:/Downloads/fig.pdf', bbox=(0,0,1,1), ylim=None):
    
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
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    plt.show()


    # Same cs ratio, different sparsity, different basis
def plot_basis_sparsity_10csratio(y1, y2, y3, y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='D:/Downloads/fig.pdf', bbox=(0,0,1,1), ylim=None):
    
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
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    plt.show()


def plotbar_cs_basis_10csratio(y1,y2,y3,y4, normalized=True, title='', labelsize=20, sizefig=(12,8), loclegend='upper left', figname='D:/Downloads/fig.pdf', bbox=(0,0,1,1), ylim=None, cspercentage=False):
    
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
    plt.savefig(figname, format='pdf', bbox_inches='tight')
    plt.show()











''' --------------------------------------------------------------- '''



if __name__ == "__main__":
    
    
    # Choose the basis to use (DWT, coif1?)
    
    plot_cs_sparsity(MtMarcyMAE_cs25, MtMarcyMAE_cs50, MtMarcyMAE_cs75, title='Reconstruction With Daubechies-2 DWT of Mt. Marcy', bbox=(1,1), figname='D:/Downloads/MtMarcyDb2CS_v7_ratio.pdf') # bbox=(0,0.925-0.0165)
    plot_cs_sparsity(DeerparkMAE_cs25, DeerparkMAE_cs50, DeerparkMAE_cs75, title='Reconstruction With Daubechies-2 DWT of Deerpark', bbox=(1,1), figname='D:/Downloads/DeerparkDb2CS_v7_ratio.pdf') # bbox=(0,0.7)
    plot_cs_sparsity(ColCircMAE_cs25, ColCircMAE_cs50, ColCircMAE_cs75, title='Reconstruction Using DCT of Columbus Circle', bbox=(1,1), figname='D:/Downloads/ColCircDCTCS_v7_ratio.pdf') # bbox=(0+0.35,0.95+0.002)
    plot_cs_sparsity(ParkslopeMAE_cs25, ParkslopeMAE_cs50, ParkslopeMAE_cs75, title='Reconstruction Using DCT of Park Slope', bbox=(1,1), figname='D:/Downloads/ParkslopeDCTCS_v7_ratio.pdf') # bbox=(0,0.9)
    
    # Choose the CS Ratio to use (25%)
    
    plot_basis_sparsity(MtMarcyMAE_dct, MtMarcyMAE_haar, MtMarcyMAE_db2, MtMarcyMAE_coif1, title='Reconstruction With 25% Compression of Mt. Marcy', bbox=(0,0.90+0.02), figname='D:/Downloads/MtMarcy25CSBases_v6.pdf')
    plot_basis_sparsity(DeerparkMAE_dct, DeerparkMAE_haar, DeerparkMAE_db2, DeerparkMAE_coif1, title='Reconstruction With 25% Compression of Deerpark', bbox=(0.65,1), figname='D:/Downloads/Deerpark25CSBases_v6.pdf')
    plot_basis_sparsity(ColCircMAE_dct, ColCircMAE_haar, ColCircMAE_db2, ColCircMAE_coif1, title='Reconstruction With 25% Compression of Columbus Circle', loclegend='lower left', figname='D:/Downloads/ColCirc25CSBases_v6.pdf')
    plot_basis_sparsity(ParkslopeMAE_dct, ParkslopeMAE_haar, ParkslopeMAE_db2, ParkslopeMAE_coif1, title='Reconstruction With 25% Compression of Park Slope', loclegend='lower left', figname='D:/Downloads/Parkslope25CSBases_v6.pdf')
    
    
    # Choose the sparsity to use (80?, 90% instead now)
    plot_cs_basis(MtMarcyMAE_dct_s, MtMarcyMAE_haar_s, MtMarcyMAE_db2_s, MtMarcyMAE_coif1_s, title='Reconstruction With 90% Sparsity of Mt. Marcy', loclegend='upper right', figname='D:/Downloads/MtMarcy90SparsityBases_v6_ratio.pdf')
    plot_cs_basis(DeerparkMAE_dct_s, DeerparkMAE_haar_s, DeerparkMAE_db2_s, DeerparkMAE_coif1_s, title='Reconstruction With 90% Sparsity of Deerpark', loclegend='upper right', figname='D:/Downloads/Deerpark90SparsityBases_v6_ratio.pdf')
    plot_cs_basis(ColCircMAE_dct_s, ColCircMAE_haar_s, ColCircMAE_db2_s, ColCircMAE_coif1_s, title='Reconstruction with 90% Sparsity of Columbus Circle', loclegend='upper right', figname='D:/Downloads/ColCirc90SparsityBases_v6_ratio.pdf')
    plot_cs_basis(ParkslopeMAE_dct_s, ParkslopeMAE_haar_s, ParkslopeMAE_db2_s, ParkslopeMAE_coif1_s, title='Reconstruction with 90% Sparsity of Park Slope', loclegend='upper right', figname='D:/Downloads/Parkslope90SparsityBases_v6_ratio.pdf')
    
    # Bar Graph
    
    plotbar_cs_basis(MtMarcyMAE_dct_s, MtMarcyMAE_haar_s, MtMarcyMAE_db2_s, MtMarcyMAE_coif1_s, title='Reconstruction With 90% Sparsity of Mt. Marcy', loclegend='upper right', figname='D:/Downloads/MtMarcy90SparsityBases_Bar_v7_ratio.pdf')
    plotbar_cs_basis(DeerparkMAE_dct_s, DeerparkMAE_haar_s, DeerparkMAE_db2_s, DeerparkMAE_coif1_s, title='Reconstruction With 90% Sparsity of Deerpark', loclegend='upper right', figname='D:/Downloads/Deerpark90SparsityBases_Bar_v7_ratio.pdf')
    plotbar_cs_basis(ColCircMAE_dct_s, ColCircMAE_haar_s, ColCircMAE_db2_s, ColCircMAE_coif1_s, title='Reconstruction With 90% Sparsity of Columbus Circle', loclegend='upper right', figname='D:/Downloads/ColCirc90SparsityBases_Bar_v7_ratio.pdf')
    plotbar_cs_basis(ParkslopeMAE_dct_s, ParkslopeMAE_haar_s, ParkslopeMAE_db2_s, ParkslopeMAE_coif1_s, title='Reconstruction With 90% Sparsity of Park Slope', loclegend='upper right', figname='D:/Downloads/Parkslope90SparsityBases_Bar_v7_ratio.pdf')
    
    
    
    ''' ----------------------------------- '''
    # ADDING 10% CS RATIO: --------------------------------------------------------------------------------------
    '''
    plot_cs_sparsity_10csratio(MtMarcyMAE_cs10, MtMarcyMAE_cs25, MtMarcyMAE_cs50, MtMarcyMAE_cs75, title='Reconstruction With Daubechies-2 DWT of Mt. Marcy', bbox=(1,1), figname='D:/Downloads/MtMarcyDb2CS_updated_v1_ratio.pdf', ylim=(0, 85))
    plot_cs_sparsity_10csratio(DeerparkMAE_cs10, DeerparkMAE_cs25, DeerparkMAE_cs50, DeerparkMAE_cs75, title='Reconstruction With Daubechies-2 DWT of Deerpark', bbox=(1,1), figname='D:/Downloads/DeerparkDb2CS_updated_v1_ratio.pdf', ylim=(0, 130))
    plot_cs_sparsity_10csratio(ColCircMAE_cs10, ColCircMAE_cs25, ColCircMAE_cs50, ColCircMAE_cs75, title='Reconstruction Using DCT of Columbus Circle', bbox=(1,1), figname='D:/Downloads/ColCircDCTCS_updated_v1_ratio.pdf', ylim=(0, 95))
    plot_cs_sparsity_10csratio(ParkslopeMAE_cs10, ParkslopeMAE_cs25, ParkslopeMAE_cs50, ParkslopeMAE_cs75, title='Reconstruction Using DCT of Park Slope', bbox=(1,1), figname='D:/Downloads/ParkslopeDCTCS_updated_v1_ratio.pdf', ylim=(0, 45))
    
    plot_cs_basis_10csratio(MtMarcyMAE_dct_s, MtMarcyMAE_haar_s, MtMarcyMAE_db2_s, MtMarcyMAE_coif1_s, title='Reconstruction With 90% Sparsity of Mt. Marcy', loclegend='upper right', figname='D:/Downloads/MtMarcy90SparsityBases_updated_v1_ratio.pdf')
    plot_cs_basis_10csratio(DeerparkMAE_dct_s, DeerparkMAE_haar_s, DeerparkMAE_db2_s, DeerparkMAE_coif1_s, title='Reconstruction With 90% Sparsity of Deerpark', loclegend='upper right', figname='D:/Downloads/Deerpark90SparsityBases_updated_v1_ratio.pdf')
    plot_cs_basis_10csratio(ColCircMAE_dct_s, ColCircMAE_haar_s, ColCircMAE_db2_s, ColCircMAE_coif1_s, title='Reconstruction with 90% Sparsity of Columbus Circle', loclegend='upper right', figname='D:/Downloads/ColCirc90SparsityBases_updated_v1_ratio.pdf')
    plot_cs_basis_10csratio(ParkslopeMAE_dct_s, ParkslopeMAE_haar_s, ParkslopeMAE_db2_s, ParkslopeMAE_coif1_s, title='Reconstruction with 90% Sparsity of Park Slope', loclegend='upper right', figname='D:/Downloads/Parkslope90SparsityBases_updated_v1_ratio.pdf')
    
    plot_basis_sparsity_10csratio(MtMarcyMAE_dct, MtMarcyMAE_haar, MtMarcyMAE_db2, MtMarcyMAE_coif1, title='Reconstruction With 10% Compression of Mt. Marcy', bbox=(0,0.90+0.02), figname='D:/Downloads/MtMarcy10CSBases_updated_v1.pdf')
    plot_basis_sparsity_10csratio(DeerparkMAE_dct, DeerparkMAE_haar, DeerparkMAE_db2, DeerparkMAE_coif1, title='Reconstruction With 10% Compression of Deerpark', bbox=(0.65,1), figname='D:/Downloads/Deerpark10CSBases_updated_v1.pdf')
    plot_basis_sparsity_10csratio(ColCircMAE_dct, ColCircMAE_haar, ColCircMAE_db2, ColCircMAE_coif1, title='Reconstruction With 10% Compression of Columbus Circle', loclegend='lower left', figname='D:/Downloads/ColCirc10CSBases_updated_v1.pdf')
    plot_basis_sparsity_10csratio(ParkslopeMAE_dct, ParkslopeMAE_haar, ParkslopeMAE_db2, ParkslopeMAE_coif1, title='Reconstruction With 10% Compression of Park Slope', loclegend='lower left', figname='D:/Downloads/Parkslope10CSBases_updated_v1.pdf')
   
    plotbar_cs_basis_10csratio(MtMarcyMAE_dct_s, MtMarcyMAE_haar_s, MtMarcyMAE_db2_s, MtMarcyMAE_coif1_s, title='Reconstruction With 90% Sparsity of Mt. Marcy', loclegend='upper right', figname='D:/Downloads/MtMarcy90SparsityBases_Bar_updated_v1_ratio.pdf')
    plotbar_cs_basis_10csratio(DeerparkMAE_dct_s, DeerparkMAE_haar_s, DeerparkMAE_db2_s, DeerparkMAE_coif1_s, title='Reconstruction With 90% Sparsity of Deerpark', loclegend='upper right', figname='D:/Downloads/Deerpark90SparsityBases_Bar_updated_v1_ratio.pdf')
    plotbar_cs_basis_10csratio(ColCircMAE_dct_s, ColCircMAE_haar_s, ColCircMAE_db2_s, ColCircMAE_coif1_s, title='Reconstruction With 90% Sparsity of Columbus Circle', loclegend='upper right',  ylim=(0,100), figname='D:/Downloads/ColCirc90SparsityBases_Bar_updated_v1_ratio.pdf')
    plotbar_cs_basis_10csratio(ParkslopeMAE_dct_s, ParkslopeMAE_haar_s, ParkslopeMAE_db2_s, ParkslopeMAE_coif1_s, title='Reconstruction With 90% Sparsity of Park Slope', loclegend='upper right',  ylim=(0,50), figname='D:/Downloads/Parkslope90SparsityBases_Bar_updated_v1_ratio.pdf')
    '''