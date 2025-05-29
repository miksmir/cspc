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
    

    
    lasfilepath = 'D:/Documents/Thesis_CS/Point_Cloud_Inputs/NYCOpenData' # Path to original file that will be downsampled
    dsfilepath = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled' # Path to write downsampled .las file to
    
    
    
    # Deerpark ---------------------------------------------------------------------------------------------
    '''
    lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las' # File name that will be downsampled # (22,370,511 points originally)
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=2.2355e-4*2, typeds='uniform', outputpath=dsfilepath, show=True) # Deerpark
    ds_array = CSPC.pcToArray(ds_path)
    trimpoints = 10000
    trim = ds_array.shape[0] - trimpoints 
    ds_array_trim = ds_array[:-trim, :]
    pc_ds_trim = CSPC.CSPCdwt()
    pc_ds_trim.setManualCoords(ds_array_trim[:,0], ds_array_trim[:,1], ds_array_trim[:,2])
    pc_ds_trim.writeLas('D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_10000ds_uniform.las')
    '''
    
    # Mt. Marcy -------------------------------------------------------------------------------------------
    
    lasfile = 'MtMarcy_u_5865088400_2015.las' # File name that will be downsampled # (17,270,767 points originally)
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=2*2.896e-4, typeds='uniform', outputpath=dsfilepath, show=True) # Mt Marcy
    ds_array = CSPC.pcToArray(ds_path)
    trimpoints = 10000
    trim = ds_array.shape[0] - trimpoints 
    ds_array_trim = ds_array[:-trim, :]
    pc_ds_trim = CSPC.CSPCdwt()
    pc_ds_trim.setManualCoords(ds_array_trim[:,0], ds_array_trim[:,1], ds_array_trim[:,2])
    pc_ds_trim.writeLas('D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_10000ds_uniform.las')
    
    
    # Park Slope -----------------------------------------------------------------------------------------
    '''
    lasfile = 'ParkSlope4thAveCarrollSt_987185_LidarClassifiedPointCloud.las' # File name that will be downsampled # (8,878,503)
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=4.75e-4, typeds='uniform', outputpath=dsfilepath, show=True) # Park Slope
    '''
    
    '''# Segmented
    lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las' # (211,686 points originally))
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=4.6e-2, typeds='uniform', outputpath=dsfilepath, show=True) # Columbus Circle
    #For 10,000 points
    #lasfile = 'test_Columbus_Circle_987217_LidarClassifiedPointCloud.las' # (303,168 points originally))
    #ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=3.3e-2, typeds='uniform', outputpath=dsfilepath, show=True) # Columbus Circle
    ds_array = CSPC.pcToArray(ds_path)
    trimpoints = 10000 # Amount of points to trim down to
    trim = ds_array.shape[0] - trimpoints 
    ds_array_trim = ds_array[:-trim, :]
    pc_ds_trim = CSPC.CSPCdwt()
    pc_ds_trim.setManualCoords(ds_array_trim[:,0], ds_array_trim[:,1], ds_array_trim[:,2])
    pc_ds_trim.writeLas('D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_10000ds_uniform.las')
    '''
    # Columbus Circle ------------------------------------------------------------------------------------------------
    '''
    lasfile = 'Columbus_Circle_987217_LidarClassifiedPointCloud.las' # File name that will be downsampled (18,487,172 points originally)
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=3.6e-4, typeds='uniform', outputpath=dsfilepath, show=True) # Columbus Circle
    '''
    
    '''# Segmented -------------------
    lasfile = 'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud.las' # (303,168 points originally))
    ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=3.3e-2, typeds='uniform', outputpath=dsfilepath, show=True) # Columbus Circle
    #For 10,000 points
    #lasfile = 'test_Columbus_Circle_987217_LidarClassifiedPointCloud.las' # (303,168 points originally))
    #ds_path, n_points = CSPC.downsample(lasfilepath, lasfile, ds_percentage=3.3e-2, typeds='uniform', outputpath=dsfilepath, show=True) # Columbus Circle
    ds_array = CSPC.pcToArray(ds_path)
    trimpoints = 10000
    trim = ds_array.shape[0] - trimpoints 
    ds_array_trim = ds_array[:-trim, :]
    pc_ds_trim = CSPC.CSPCdwt()
    pc_ds_trim.setManualCoords(ds_array_trim[:,0], ds_array_trim[:,1], ds_array_trim[:,2])
    pc_ds_trim.writeLas('D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_10000ds_uniform.las')
    '''
    
    
    
    
    
    
    '''
    trim = abs(pc_ds.shape[0] - npoints_to_trim_to)     # HAS to be positive (trim = pc_ds.shape[0] > npoints_to_trim_to)
    pc_ds_trim = pc_ds[:-trim, :]
    
        i.e.:
            2 = abs(5002 - 5000)
            pc_ds_trim = pc_ds[:-2, :]
    '''
    
    # ds_percentage * total_points MUST BE an odd value for CSPC.downsample() to work