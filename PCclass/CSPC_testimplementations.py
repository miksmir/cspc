# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:20:42 2025

@author: Mikhail
"""
import CSPointCloud as CSPC
# ----------------------------------------------------------------
if __name__ == "__main__":
    print("hello world")


# Chamfer Distance, Hausdorf Distance, Earth Mover's Distance
    import point_cloud_utils as pcu
    pc1 = CSPC.pcToArray("D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\reconstruction\\NYCOpenData\\thresholding\\MtMarcy\\MtMarcy_u_5865088400_2015_90thresholded_DWT_Db2_75Gaussian_of_5000ds_uniform.las")
    pc2 = CSPC.pcToArray("D:\\Documents\\Thesis_CS\\Point_Cloud_Outputs\\reconstruction\\NYCOpenData\\thresholding\\MtMarcy\\MtMarcy_u_5865088400_2015_90thresholded_DWT_Db2_25Gaussian_of_5000ds_uniform.las")

    cd = pcu.chamfer_distance(pc1, pc2)
    hd = pcu.hausdorff_distance(pc1, pc2)
    emd, pi = pcu.earth_movers_distance(pc1, pc2)
    
    print(f"Chamfer Distance: {cd}")
    print(f"Hausdorf Distance: {hd}")
    print(f"Earth Mover's Distance:  {emd}")
    
    