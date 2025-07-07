import os
import CSPointCloud as CSPC
from CSPC_simfunc import runCSPCdwt, runCSPCdct, runCSPCdft, runCSPCdct2
#from CSPointCloud import INPUT_PATH_PCLAS, OUTPUT_PATH_PCLAS

if __name__ == "__main__":
    
    #arr_sparsity_val = [70,80,90]
    #arr_cs_ratios = [0.75, 0.5, 0.25]
    
    # arr_sparsity_val = [10,20,30,40,50,60,70,80,90]
    # arr_cs_ratios = [0.75]
    
    arr_sparsity_val = [90]
    arr_cs_ratios = [0.05]
    
    for cs_ratio in arr_cs_ratios:
        for sparsity_val in arr_sparsity_val:
            runCSPCdwt(inputpc = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', outputname = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', wvlt = 'db2', ds_type='uniform', parallel=True)
            runCSPCdct(inputpc = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud_5000ds_uniform.las', outputname = 'ParkSlope4thAveCarrollSt_987185_Buildings_LidarClassifiedPointCloud.las', pcname = 'Park Slope', cs_ratio=cs_ratio, sparsity_val=sparsity_val, measurement_type='gaussian', ds_type='uniform', parallel=True)
   