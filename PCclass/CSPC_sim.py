# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 12:36:24 2025

@author: misha
"""


import CSPointCloud as CSPC
from CSPC_simfunc import runCSPCdwt, runCSPCdct, runCSPCdft, runCSPCdct2
import point_cloud_utils as pcu

if __name__ == "__main__":
    
    #arr_sparsity_val = [70,80,90]
    #arr_cs_ratios = [0.75, 0.5, 0.25]
    
    #arr_sparsity_val = [40,50,60,70,80,90]
    #arr_cs_ratios = [0.25, 0.50, 0.75]
    
    arr_sparsity_val = [90]
    arr_cs_ratios = [0.05]
    
    for cs_ratio in arr_cs_ratios:
        for sparsity_val in arr_sparsity_val:
            runCSPCdwt(inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'db2', ds_type='uniform', parallel=True)
            runCSPCdct(inputpath = 'D:/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='D:/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
    
    #arr_sparsity_val = [90]
    #arr_cs_ratios = [0.10]
    
    
    #for cs_ratio in arr_cs_ratios:
    #    for sparsity_val in arr_sparsity_val:
            
            #runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_10000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'haar', ds_type='uniform', parallel=True)
            #runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/MtMarcy', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mt Marcy', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'haar', ds_type='uniform', parallel=True)
            #runCSPCdct(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/MtMarcy', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mt Marcy', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
            #runCSPCdct(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_10000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
            #runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_10000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las', pcname = 'Deerpark', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'db2', ds_type='uniform', parallel=True)
            #runCSPCdct(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_10000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las', pcname = 'Deerpark', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
           
    #for cs_ratio in arr_cs_ratios:
     #   for sparsity_val in arr_sparsity_val:
      #        runCSPCdwt(inputpath = 'D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\downsampled\\Deerpark_Cuddebackville_u_5325059000_2022_5000ds_uniform.las', path='D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\reconstruction\\NYCOpenData\\thresholding', lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las', pcname = 'Deerpark', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'haar', ds_type='uniform')
              #runCSPCdct(inputpath = 'D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\downsampled\\Deerpark_Cuddebackville_u_5325059000_2022_5000ds_uniform.las', path='D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\reconstruction\\NYCOpenData\\thresholding', lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las', pcname = 'Deerpark', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform')
    
    #runCSPCdct2(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mount Marcy', cs_ratio=0.10, sparsity_val=90, measurement_type='gaussian', ds_type='uniform')
    
    
    #runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mount Marcy', cs_ratio=0.05, sparsity_val=90, measurement_type='gaussian', wvlt = 'haar', ds_type='uniform')
    #runCSPCdct(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mount Marcy', cs_ratio=0.05, sparsity_val=90, measurement_type='gaussian', ds_type='uniform')
    #runCSPCdft(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mount Marcy', cs_ratio=0.05, sparsity_val=90, measurement_type='gaussian', ds_type='uniform')
    
    # LATER with Chamfer Distance and with 10% Reconstruction:
        # arr_sparsity_val = [10,20,30,40,50,60,70,80,90]
        # arr_cs_ratios = [0.10]
        # rundwt(haar) Mt Marcy
        # rundwt(haar) Deerpark
        # rundwt(haar) Columbus Circle
        # rundwt(haar) Park Slope
        # rundwt(db2) Mt Marcy
        # rundwt(db2) Deerpark
        # rundwt(db2) Columbus Circle
        # rundwt(db2) Park Slope
        # rundwt(coif1) Mt Marcy
        # rundwt(coif1) Deerpark
        # rundwt(coif1) Columbus Circle
        # rundwt(coif1) Park Slope
        # rundct() Mt Marcy
        # rundct() Deerpark
        # rundct() Columbus Circle
        # rundct() Park Slope
        
    '''    
    arr_sparsity_val = [10,20,30,40,50,60,70,80,90]
    arr_cs_ratios = [0.10]
    
    for cs_ratio in arr_cs_ratios:
        for sparsity_val in arr_sparsity_val:
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/MtMarcy', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mt Marcy', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'haar', ds_type='uniform', parallel=True)
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/MtMarcy', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mt Marcy', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'db2', ds_type='uniform', parallel=True)
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/MtMarcy', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mt Marcy', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'coif1', ds_type='uniform', parallel=True)
            runCSPCdct(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/MtMarcy_u_5865088400_2015_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/MtMarcy', lasfile = 'MtMarcy_u_5865088400_2015.las', pcname = 'Mt Marcy', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
            
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/Deerpark', lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las', pcname = 'Deerpark', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'haar', ds_type='uniform', parallel=True)
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/Deerpark', lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las', pcname = 'Deerpark', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'db2', ds_type='uniform', parallel=True)
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/Deerpark', lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las', pcname = 'Deerpark', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'coif1', ds_type='uniform', parallel=True)
            runCSPCdct(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Deerpark_Cuddebackville_u_5325059000_2022_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/Deerpark', lasfile = 'Deerpark_Cuddebackville_u_5325059000_2022.las', pcname = 'Deerpark', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
            
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/ColumbusCircle', lasfile = 'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud.las', pcname = 'Columbus Circle', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'haar', ds_type='uniform', parallel=True)
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/ColumbusCircle', lasfile = 'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud.las', pcname = 'Columbus Circle', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'db2', ds_type='uniform', parallel=True)
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/ColumbusCircle', lasfile = 'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud.las', pcname = 'Columbus Circle', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'coif1', ds_type='uniform', parallel=True)
            runCSPCdct(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/ColumbusCircle', lasfile = 'Columbus_Circle_987217_Buildings_LidarClassifiedPointCloud.las', pcname = 'Columbus Circle', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
            
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/ParkSlope', lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'haar', ds_type='uniform', parallel=True)
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/ParkSlope', lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'db2', ds_type='uniform', parallel=True)
            runCSPCdwt(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/ParkSlope', lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'coif1', ds_type='uniform', parallel=True)
            runCSPCdct(inputpath = 'C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/downsampled/ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', path='C:/Users/misha/Documents/Thesis_CS/Point_Cloud_Outputs/reconstruction/NYCOpenData/thresholding/ParkSlope', lasfile = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
        '''
                    
        
        