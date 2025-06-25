# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:24:06 2025

@author: Mikhail
"""


import CSPointCloud as CSPC
import numpy as np
import os

if __name__ == "__main__":
    """ Creating and exporting a downsampled point cloud example """
    
    lasfilepath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Inputs/NYCOpenData' # Path to original file that will be downsampled
    dsfilepath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled' # Path to write downsampled .las file to
    '''
    lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # File name that will be downsampled
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=3e-4, typeds='uniform', outputpath=dsfilepath, show=True) # Deerpark
    
    lasfile = 'MtMarcy_u_5865088400_2015.las' # File name that will be downsampled
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=6.5e-4, typeds='uniform', outputpath=dsfilepath, show=True) # Mt Marcy
    
    lasfile = 'ParkSlope4thAveCarrollSt_987185_LidarClassifiedPointCloud.las' # File name that will be downsampled
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=4.75e-4, typeds='uniform', outputpath=dsfilepath, show=True) # Park Slope
    
    lasfile = 'Columbus_Circle_987217_LidarClassifiedPointCloud.las' # File name that will be downsampled
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=3.6e-4, typeds='uniform', outputpath=dsfilepath, show=True) # Columbus Circle
    '''
    
    lasfile = 'WallSt_980195_LidarClassifiedPointCloud.las'
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=3e-4, typeds='uniform', outputpath=dsfilepath, show=True) # Wall Stret
    
    
    
    
    
    # ds_percentage * total_points MUST BE an odd value for CSPC.downsample() to work